# Databricks notebook source
# MAGIC %md
# MAGIC ### Model Monitoring

# COMMAND ----------

# MAGIC %md
# MAGIC - Use the training and inference date widgets to highlight model performance on unseen data

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC - Specify your criteria for retraining and promotion to production.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC - Use common model performance visualizations to highlight the performance of Staged Model vs. Production Model. E.g. residual plot comparisons

# COMMAND ----------

def score_model(data, model_uri):
    model = mlflow.sklearn.load_model(model_uri)
    return model.predict(data)

# COMMAND ----------

# Load the score data
score_path = "/dbfs/FileStore/shared_uploads/lloyd.palum@rochester.edu/score_windfarm_data.csv"
score_df = Utils.load_data(score_path, index_col=0)
score_df.head()

# COMMAND ----------

# Drop the power column since we are predicting that value
actual_power = pd.DataFrame(score_df.power.values, columns=['power'], index=score_df.index)
score = score_df.drop("power", axis=1)
actual_power.head()

# COMMAND ----------

# Formulate the model URI to fetch from the model registery
model_uri = "models:/{}/{}".format('sk-learn-PowerForecastingModel', 'Production')

# Predict the Power output 
pred_production = pd.DataFrame(score_model(score, model_uri), columns=["predicted_production"], index=actual_power.index)
pred_production

actual_power["predicted_production"] = pred_production["predicted_production"]
actual_power

# COMMAND ----------

# Formulate the model URI to fetch from the model registery
model_uri = "models:/{}/{}".format('sk-learn-PowerForecastingModel', 'Staging')

# Predict the Power output
pred_staging = pd.DataFrame(score_model(score, model_uri), columns=["predicted_staging"], index=actual_power.index)
pred_staging

actual_power["predicted_staging"] = pred_staging["predicted_staging"]
actual_power

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC import matplotlib.pyplot as plt
# MAGIC actual_power.plot.line(figsize=(11,8))
# MAGIC plt.title("Production and Staging Model Comparison")
# MAGIC plt.ylabel("Mega-Watts")
# MAGIC plt.xlabel("Day of the Year")

# COMMAND ----------

logged_model = 'runs:/5cc37f8fe36b48af8b38117a869103aa/model'

# Load model as a Spark UDF.
loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model)

# grab the feature vector columns
columns = predictions.columns[:-4]

# Predict on a Spark DataFrame.
results = predictions.withColumn('sklearn_prediction', loaded_model(*columns))
results = results.withColumn('mllib_residual', (F.col("cnt") - F.col("prediction")))
results = results.withColumn('sklearn_residual', (F.col("cnt") - F.col("sklearn_prediction")))
results = results.withColumnRenamed("prediction", "mllib_prediction")

# COMMAND ----------

# Residual Plot
tmp = results.select("hr","mllib_residual","sklearn_residual").toPandas().groupby('hr').mean()
tmp.plot.bar(figsize=(11,8), title="Average MLLIB GBT residual {:4.2f}     Average SKlearn GBT residual {:4.2f}".format(tmp['mllib_residual'].mean(),tmp['sklearn_residual'].mean()))

# COMMAND ----------

# MAGIC %md
# MAGIC - Include code that allows the monitoring notebook to “promote” a model from staging to production.

# COMMAND ----------

from mlflow.tracking.client import MlflowClient
from mlflow.entities import ViewType

client = MlflowClient()
# Parametrizing the right experiment path using widgets
experiment_name = 'Default'
experiment = client.get_experiment_by_name(experiment_name)
experiment_ids = [experiment.experiment_id]
print("Experiment IDs:", experiment_ids)

# Setting the decision criteria for a best run
query = "metrics.accuracy > 0.8"
runs = client.search_runs(experiment_ids, query, ViewType.ALL)

# Searching throught filtered runs to identify the best_run and build the model URI to programmatically reference later
accuracy_high = None
best_run = None
for run in runs:
  if (accuracy_high == None or run.data.metrics['accuracy'] > accuracy_high):
    accuracy_high = run.data.metrics['accuracy']
    best_run = run
run_id = best_run.info.run_id
print('Highest Accuracy: ', accuracy_high)
print('Run ID: ', run_id)

model_uri = "runs:/" + run_id + "/model"

# COMMAND ----------

# if staging RMSE < production RMSE --> execute next cell and promote staging to production
# else retrain staging model

# COMMAND ----------

import time

# Check if model is already registered
model_name = "News Classification Model"
try:
  registered_model = client.get_registered_model(model_name)
except:
  registered_model = client.create_registered_model(model_name)

# Create the model source
model_source = f"{best_run.info.artifact_uri}/model"
print(model_source)

# Archive old production model
max_version = 0
for mv in client.search_model_versions("name='Diabetes Progression Model'"):
  current_version = int(dict(mv)['version'])
  if current_version > max_version:
    max_version = current_version
  if dict(mv)['current_stage'] == 'Production':
    version = dict(mv)['version']
    client.transition_model_version_stage(model_name, version, stage='Archived')

# Create a new version for this model with best metric (accuracy)
client.create_model_version(model_name, model_source, run_id)
# Check the status of the created model version (it has to be READY)
status = None
while status != 'READY':
  for mv in client.search_model_versions(f"run_id='{run_id}'"):
    status = mv.status if int(mv.version)==max_version + 1 else status
  time.sleep(5)

# Promote the model version to production stage
client.transition_model_version_stage(model_name, max_version + 1, stage='Production')