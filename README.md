# DSCC202-402-Forecasting-Flight-Delay-Final-Project

### Coauthors:
- Jordan Pappas, University of Rochester

### Abstract:
Data intensive applications (DIA) are an important part of many valuable services that we rely on in our day to day lives. These applications in most cases are built by performing data engineering and data science at scale. Scale in this case implies data volume and compute capacity far outside of what is available on a single machine and its narrow connection to the internet.

To this end, we developed a scalable machine learning product to predict flight delays and rank airline traffic in real time using flight record data and hourly weather report data. By collaborating with data engineers and data scientists, our team built a end-to-end ML pipeline using ETL, exploratory data analysis, MLOps, and model monitoring.



# Preview

![](https://github.com/jordanjpappas/Portfolio/blob/master/images/O%26G-cluster_maps.png)



# Usage
[(Back to top)](#table-of-contents)

Scripts:
* Analysis is run in the order shown below.
    - init-jordan.R: Load packages and set directory hidden objects
    - 00-collectwelldata.R: Import drilling data, filter by production types (i.e. 'OIL & GAS','GAS','OIL','GAS OR COALBED','CBM'), and aggregate data to county-year level
    - 01-convert_BOE.R: Convert drilling data to barrels of oil (BOE) production data
    - 1a-run_mixedmodels.R: Apply cluster finite mixture models to drilling data and plot various time-series cluster model outputs
    - 1b-run_mixedmodels_BOE.R: Apply cluster finite mixture models to production data and plot various time-series cluster model outputs
    - 2a-plot_sum_of_squares.py: Calculate sum of squares measure for various cluster model outputs and plot scree visualization
    - 2b-map_states.R: Plot various state geography cluster model outputs
    - 2c-map_shale_plays.R: Plot various shale play geography cluster model outputs
    - 3-calculate_summ_stats.py: Calculate key variable summary statistics
    - 4a-construct_county_shares.R: Construct state-level dataset that counts number of counties in each cluster by state
    - 4b-construct_play_shares.R: Construct play-level dataset that counts number of counties in each cluster by state
    - 5-compare_weighted_unweighted_approach.R: Compare unweighted and weighted approach cluster results
    - 6-compare_data_distributions.R: Create dataset of counties with a categorical variable for drilling vs. production data missingness and plot state geography cluster missigness results
    - 99-compile-docx.R: Compile previous script outputs into Microsoft Word .docx format
