# DSCC202-402-Forecasting-Flight-Delay-Final-Project

### Coauthors:
- Jordan Pappas, University of Rochester

### Abstract:
The last two decades have witnessed revolutionary changes in America’s upstream oil and gas industry, with a meteoric rise in total production and a shift in where this production takes place. Many researchers have used the shift in the geographic location of drilling to study the effects of oil and gas development on local communities. In this paper we seek, to the extent possible, a stylized representation of this recent geographic shift. Specifically, we ask to what extent the timing of drilling across counties in the contiguous United States over the last two decades can be explained by places of old drilling versus places of new drilling, or if there are significant shares of additional categories of counties, e.g. places that have experienced both.

To this end, we employ finite mixture models fit to data on county-level drilling shares by year. We find that two clusters reduce the sum of squared errors by about one third, while third and fourth clusters add only 7.5 and 5 percentage points additional explanatory power, respectively. Our main takeaway is that counties can be categorized into old versus new drilling, with relatively little room for additional typologies. Our approach allows us to spatially identify old drilling and new drilling counties, which we map and compare with well-known oil and gas fields.



# Preview

![](https://github.com/jordanjpappas/Portfolio/blob/master/images/O%26G-cluster_maps.png)
Figure 2: Maps of Drilling Counties by Cluster. Rows vary by the number of clusters in the model: the topmost row shows the one-cluster model, the middle row shows the two-cluster model, and the last row shows the three-cluster model. Columns vary by the cluster in which each county is classified. Thus, the first plot (top-left corner) highlights all counties in our analytic sample. The second row of plots splits counties into two clusters: early drilling (the left plot) and late drilling (the right plot). The third-row splits counties into three clusters: early drilling, middle drilling, and late drilling. Clusters are always ordered by the “peak year,” i.e. year of the maximum parameter.



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
