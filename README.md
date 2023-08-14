# Summer Coastal Fog in Washington through GOES-17 satellite data and field photos

This repository contains the code for the research project ***Summer Fog Frequency Patterns and Impact on Intertidal Organisms around Washington Coast from GOES-17 Satellite Imagery, Field Photos, and Field Sensors*** by Autumn Nguyễn, Jessica Lundquist, Steven Pestana, and Eli Schwat (2023).

<img src="https://drive.google.com/uc?export=view&id=1-kkUkIVnFT0kWfjuF7QpftC-sU8bkgQB" style="height:500px"> <t> <img alt="Screen Shot 2023-08-13 at 3 37 32 PM" src="https://github.com/autumn-yng/summerfog/assets/92401509/b0c9c6ae-2682-45fd-bc17-8247722ef8a4" style="height:300px">

## Research summary
Fog can offer protection for intertidal organisms around Washington coast during hot low tide times, which is especially important when more extreme heatwaves are likely to happen due to global warming. Using the Cloud Top Height imagery from NASA and NOAA’s satellite GOES-17, we created fog and low clouds (**FLC**) frequency maps and FLC timeseries for the summer months (May through September) of 2022 in Washington coastal areas. We also deployed cameras to take pictures of areas on San Juan Island and used a Computer Vision model to classify the pictures as having fog or not having fog. Our main research goals were: 
1. Finding the frequency patterns of FLC in the summer around the Washington coast using GOES-17 Cloud Top Height data  
2. Comparing the cloud top height data with the field cameras’ photos and the sensor’s temperature and humidity data, to see how we can best interpret the cloud top height data  
3. Quantifying the impact of FLC on intertidal organisms in Washington by the number of FLC-protected hours during midday low tide

## Which code does which purposes
### Main code:
- We loaded, stacked, and **chunked** all the raster files, and saved them in a **zarr** format in [goes/goes_create_zarr.ipynb](goes/goes_create_zarr.ipynb).

<img src="https://drive.google.com/uc?export=view&id=1KFBGF-bETGhJ0mJPusuQaZZHko4KfrG-" style="width:500px">

- We created **frequency maps** in [goes/goes_frequency.ipynb](goes/goes_frequency.ipynb)

<img src="https://drive.google.com/uc?export=view&id=1VvqlYuMHauPtT2kBeeiZwnIs8Hg64249">

- We calculated FLC frequency and ploted Cloud Top Height **time series of individual pixels** in [goes/goes_analyze_timeseries.ipynb](goes/goes_analyze_timeseries.ipynb)

- We **classified** field photos using a **Machine Learning** model in [img_classification.ipynb](img_classification.ipynb)

- We **compared** satellite-detected cloud height and camera-derived fog presence in [compare-goes-vs-photos.ipynb](compare-goes-vs-photos.ipynb)

<img src="https://drive.google.com/uc?export=view&id=1kQ2swG9nwjpIcr7LRRE2pXoivcNNrimS">

- We calculated the number of hours that intertidal organisms had protection from FLC during **midday low tide** in [tide-vs-cloudheight.ipynb](tide-vs-cloudheight.ipynb)

<img src="https://drive.google.com/uc?export=view&id=16mCcI0uQ0Zq2HKVHmHWm5Eap6BsKFcLd">

### Supplemental code:
- Our guide for downloading raster files from GOES-17: [archived/getting-started.ipynb](archived/getting-started.ipynb)

- Where we added time index and remove unnecessary coordinates and dimensions in the raster files to reduce size (disk usage): [goes/goes_add_time_and_clean.ipynb](goes/goes_add_time_and_clean.ipynb)

- Our code for plotting the shoreline in the frequency maps: [plot_shoreline.py](plot_shoreline.py)

- The environment we coded in: [environment.yml](environment.yml).
We created the environment in Terminal using `conda env create -f environment.yml`.


