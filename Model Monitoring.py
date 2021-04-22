# Databricks notebook source
# MAGIC %md
# MAGIC ## Model Monitoring

# COMMAND ----------

# MAGIC %run ./includes/utilities

# COMMAND ----------

# MAGIC %run ./includes/configuration

# COMMAND ----------

# grab the station information (system wide)
stationDF=get_bike_stations()[['name','station_id','lat','lon']]

# grab the stations of interest
stationsOfInterestDF = spark.sql("""select distinct(station_id) from from citibike.forecast_regression_timeweather;""").toPandas()
stationDF = stationDF[stationDF['station_id'].apply(lambda x: int(x) in list(stationsOfInterestDF.values.flatten()))]

# COMMAND ----------

from datetime import datetime as dt
from datetime import timedelta

dbutils.widgets.removeAll()

dbutils.widgets.dropdown("00.Airport_Code", "JFK", ["JFK","SEA","BOS","ATL","LAX","SFO","DEN","DFW","ORD","CVG","CLT","DCA","IAH"])
dbutils.widgets.text('01.training_start_date', "2018-01-01")
dbutils.widgets.text('02.training_end_date', "2019-03-15")
dbutils.widgets.text('03.inference_date', (dt.strptime(str(dbutils.widgets.get('02.training_end_date')), "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d"))
dbutils.widgets.text('05.promote_model', "No")

training_start_date = str(dbutils.widgets.get('01.training_start_date'))
training_end_date = str(dbutils.widgets.get('02.training_end_date'))
inference_date = str(dbutils.widgets.get('03.inference_date'))
airport_code = str(dbutils.widgets.get('00.Airport_Code'))

if dbutils.widgets.get("05.promote_model")=='Yes':
  promote_model = True
else:
  promote_model = False
  
print(airport_code,training_start_date,training_end_date,inference_date,promote_model)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Forecast flight delay at selected airport

# COMMAND ----------

import mlflow
from pprint import pprint
from mlflow.tracking import MlflowClient
import plotly.express as px
from datetime import timedelta, datetime

client = MlflowClient()

# COMMAND ----------

# assemble dataset for forecasting
fdf = spark.sql('''
   SELECT
    a.hour as ds,
    EXTRACT(year from a.hour) as year,
    EXTRACT(dayofweek from a.hour) as dayofweek,
    EXTRACT(hour from a.hour) as hour,
    CASE WHEN d.date IS NULL THEN 0 ELSE 1 END as is_holiday,
    COALESCE(c.tot_precip_mm,0) as precip_mm,
    c.avg_temp_f as temp_f
  FROM ( -- all rental hours by currently active stations
    SELECT 
      y.station_id,
      x.hour
    FROM citibike.periods x
    INNER JOIN citibike.stations_most_active y
     ON x.hour BETWEEN '{0}' AND '{1}'
    ) a
  LEFT OUTER JOIN citibike.rentals b
    ON a.station_id=b.station_id AND a.hour=b.hour
  LEFT OUTER JOIN citibike.weather c
    ON a.hour=c.time
  LEFT OUTER JOIN citibike.holidays d
    ON TO_DATE(a.hour)=d.date
  WHERE a.station_id = '{2}'
  '''.format(end_date, (datetime.strptime(end_date, '%Y-%m-%d') + timedelta(hours=int(hours_to_forecast))).strftime("%Y-%m-%d %H:%M:%S"), station_id)
  )

# COMMAND ----------

# Forecast using the production and staging models

df1=fdf.toPandas().fillna(method='ffill').fillna(method='bfill')
df1['model']='Production'
df1['yhat']=prod_model.predict(df1.drop(["ds","model"], axis=1).values)

df2=fdf.toPandas().fillna(method='ffill').fillna(method='bfill')
df2['model']='Staging'
df2['yhat']=stage_model.predict(df2.drop(["ds","model"], axis=1).values)

# COMMAND ----------

df = pd.concat([df1,df2]).reset_index()
labels={
   "ds": "Forecast Time",
   "yhat": "Forecasted Delay",
   "model": "Model Stage"
}
fig = px.line(df, x="ds", y="yhat", color='model', title=f"{airport_code} delay forecast by model stage", labels=labels)
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Monitoring the model performance

# COMMAND ----------

train_df = spark.sql('''
   SELECT
    a.hour as ds,
    EXTRACT(year from a.hour) as year,
    EXTRACT(dayofweek from a.hour) as dayofweek,
    EXTRACT(hour from a.hour) as hour,
    CASE WHEN d.date IS NULL THEN 0 ELSE 1 END as is_holiday,
    COALESCE(c.tot_precip_mm,0) as precip_mm,
    c.avg_temp_f as temp_f
  FROM ( -- all rental hours by currently active stations
    SELECT 
      y.station_id,
      x.hour
    FROM citibike.periods x
    INNER JOIN citibike.stations_most_active y
     ON x.hour BETWEEN '{0}' AND '{1}'
    ) a
  LEFT OUTER JOIN citibike.rentals b
    ON a.station_id=b.station_id AND a.hour=b.hour
  LEFT OUTER JOIN citibike.weather c
    ON a.hour=c.time
  LEFT OUTER JOIN citibike.holidays d
    ON TO_DATE(a.hour)=d.date
  WHERE a.station_id = '{2}'
  '''.format((datetime.strptime(end_date, '%Y-%m-%d') - timedelta(hours=int(hours_to_forecast))).strftime("%Y-%m-%d %H:%M:%S"), end_date,  station_id)
  )

# COMMAND ----------

airport = dbutils.widgets.get('00.Airport_Code')
airport_id = stationDF[stationDF['name']==airport]['station_id'].values[0]

model_name = "{}-reg-rf-model".format(airport_id)

prod_version = None
stage_version = None
# get the respective versions
for mv in client.search_model_versions(f"name='{model_name}'"):
  if dict(mv)['current_stage'] == 'Staging':
    stage_version=dict(mv)['version']
  elif dict(mv)['current_stage'] == 'Production':
    prod_version=dict(mv)['version']

if prod_version is not None:
  # load the training data associated with the production model
  prod_model = mlflow.sklearn.load_model(f"models:/{model_name}/Production")
  pdf = spark.sql(f"""SELECT * from citibike.forecast_regression_timeweather WHERE station_id = '{station_id}' and model_version = '{prod_version}';""").toPandas()
if stage_version is not None:
  # load the training data assocaited with the staging model
  stage_model = mlflow.sklearn.load_model(f"models:/{model_name}/Staging")
  sdf = spark.sql(f"""SELECT * from citibike.forecast_regression_timeweather WHERE station_id = '{station_id}' and model_version = '{stage_version}';""").toPandas()

# COMMAND ----------

pdf['stage']="prod"
pdf['residual']=pdf['y']-pdf['yhat']

sdf['stage']="staging"
sdf['residual']=sdf['y']-sdf['yhat']

df=pd.concat([pdf,sdf])

# COMMAND ----------

fig = px.scatter(
    df, x='yhat', y='residual',
    marginal_y='violin',
    color='stage', trendline='ols',
    title=f"{airport} delay forecast model performance comparison"
)
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Use Tensorflow Validation Library
# MAGIC - check schema between the training and serving periods of time
# MAGIC - check for data drift and skew between training and serving

# COMMAND ----------

from sklearn.model_selection import train_test_split
import tensorflow_data_validation as tfdv
from tensorflow_data_validation.utils.display_util import get_statistics_html
import warnings

warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

stats_train=tfdv.generate_statistics_from_dataframe(dataframe=train_df.toPandas())
stats_serve=tfdv.generate_statistics_from_dataframe(dataframe=fdf.toPandas())

schema = tfdv.infer_schema(statistics=stats_train)
tfdv.display_schema(schema=schema)

# COMMAND ----------

# Compare evaluation data with training data
displayHTML(get_statistics_html(lhs_statistics=stats_serve, rhs_statistics=stats_train,
                          lhs_name='SERVE_DATASET', rhs_name='TRAIN_DATASET'))

# COMMAND ----------

anomalies = tfdv.validate_statistics(statistics=stats_serve, schema=schema)
tfdv.display_anomalies(anomalies)

# COMMAND ----------

# Add skew and drift comparators
temp_f = tfdv.get_feature(schema, 'temp_f')
temp_f.skew_comparator.jensen_shannon_divergence.threshold = 0
temp_f.drift_comparator.jensen_shannon_divergence.threshold = 0

precip_mm = tfdv.get_feature(schema, 'precip_mm')
precip_mm.skew_comparator.jensen_shannon_divergence.threshold = 0
precip_mm.drift_comparator.jensen_shannon_divergence.threshold = 0

_anomalies = tfdv.validate_statistics(stats_train, schema, serving_statistics=stats_serve)

hour = tfdv.get_feature(schema, 'hour')
hour.skew_comparator.jensen_shannon_divergence.threshold = 0
hour.drift_comparator.jensen_shannon_divergence.threshold = 0

dayofweek = tfdv.get_feature(schema, 'dayofweek')
dayofweek.skew_comparator.jensen_shannon_divergence.threshold = 0
dayofweek.drift_comparator.jensen_shannon_divergence.threshold = 0

_anomalies = tfdv.validate_statistics(stats_train, schema, serving_statistics=stats_serve)

tfdv.display_anomalies(_anomalies)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Promote model if selected

# COMMAND ----------

# promote staging to production
if promote_model and stage_version is not None and prod_version is not None:

  # Archive the production model
  client.transition_model_version_stage(
      name=model_name,
      version=prod_version,
      stage="Archived"
  )

  # Staging --> Production
  client.transition_model_version_stage(
      name=model_name,
      version=stage_version,
      stage="Production"
  )

# COMMAND ----------

import json

# Return Success
dbutils.notebook.exit(json.dumps({"exit_code": "Success"}))