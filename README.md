# Summer Coastal Fog in Washington through GOES-17 satellite data and field photos

This repository contains the code for the research project ***Summer Fog Frequency Patterns and Impact on Intertidal Organisms around Washington Coast from GOES-17 Satellite Imagery, Field Photos, and Field Sensors*** by Autumn Nguyễn, Jessica Lundquist, Steven Pestana, and Eli Schwat (2023).

## Research summary
Fog can offer protection for intertidal organisms around Washington coast during hot low tide times, which is especially important when more extreme heatwaves are likely to happen due to global warming. Using the Cloud Top Height imagery from NASA and NOAA’s satellite GOES-17, we created fog and low clouds (**FLC**) frequency maps and FLC timeseries for the summer months (May through September) of 2022 in Washington coastal areas. We also deployed cameras to take pictures of areas on San Juan Island and used a Computer Vision model to classify the pictures as having fog or not having fog. Our main research goals were: 
1. Finding the frequency patterns of FLC in the summer around the Washington coast using GOES-17 Cloud Top Height data  
2. Comparing the cloud top height data with the field cameras’ photos and the sensor’s temperature and humidity data, to see how we can best interpret the cloud top height data  
3. Quantifying the impact of FLC on intertidal organisms in Washington in terms of FLC-protected hours during midday low tide and of the temperature difference when there is FLC and when there is not

## Which code does which purposes

- We loaded, stacked, and chunked all the raster files, and saved them in a zarr format in [goes/goes_create_zarr.ipynb](goes/goes_create_zarr.ipynb).

<img src="https://drive.google.com/uc?export=view&id=1KFBGF-bETGhJ0mJPusuQaZZHko4KfrG-" style="width:500px">

- We created frequency maps in [goes/goes_frequency.ipynb](goes/goes_frequency.ipynb)

<img src="https://drive.google.com/uc?export=view&id=1-kkUkIVnFT0kWfjuF7QpftC-sU8bkgQB" style="width:500px">

- We calculated FLC frequency and ploted Cloud Top Height time series of individual pixels in [goes/goes_analyze_timeseries.ipynb](goes/goes_analyze_timeseries.ipynb)

- We compared satellite-detected cloud height and camera-derived fog presence in 


