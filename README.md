# The Winter Bottleneck: Analyzing Dock Capacity Pressure in Boston’s Bluebikes System
This project explores the operational bottlenecks of Boston’s Bluebikes system during severe winter weather. 
By combining November 2025 trip history, station metadata, and Open-Meteo hourly weather data, 
this project aims to analyze how snow and freezing temperatures trigger imbalanced one-way trips.
As result, this project aims to identify which districts and stations experience the highest theoretical
overflow pressure during different weather, providing insights for optimizing rebalancing dispatch.

At the current stage, the project has completed the data retrieve and loading process. The code can now
automatically download and store trip and station data, while also retrieve hourly historical weather 
data through the Open-Meteo API.

# Data sources
1. Bluebikes Trip history
- Name: 202511-bluebikes-tripdata.csv.zip 
- URL: https://s3.amazonaws.com/hubway-data/index.html
- Type: File 
- List of fields: Ride_id, Rideable_type, Started_at, Ended_at, Start_station_name, Start_station_id, End_station_name, End_station_id, Start_lat, Start_ln, End_lat, End_ln, member_casual 
- Formate: CSV
- Estimated data size: 344,842 trips
2. Bluebike Station data
- Name: -External-_11.17.25_Bluebikes_Station_List.csv
- URL: https://bluebikes.com/system-data
- Type: File
- List of fields: Number,	NAME, Lat, Long, Seasonal Status, Municipality, Total Docks, Station ID 
- Formate: CSV
- Estimated data size: 597 stations
3. Weather data
- Name: Open-Meteo Hourly Weather Data
- URL: https://open-meteo.com/en/docs/historical-weather-api?start_date=2025-01-01&end_date=2025-01-31
- Type: API
- List of fields: 'data', 'temperature_2m', 'precipitation', 'snowfall', 'wind_speed_10m', 'cloud_cover' 
- Formate: JSON
- Estimated data size: ~4,464 data points

# Results 
At this current stage, no direct analytical finding result is available.
Current progress: 
1. automated download of trip history data
2. automated download of station data
3. retrieve hourly historical weather data from open-meteo API
4. clean and standardized the data of trip, station, and weather
5. create the hourly station bike flow data
6. calculated the preliminary station pressure index based on next flow and total doc capacity
7. create the merged dataset containing station, time, pressure index, and weather information, ready to be used for analysis

# Installation
API: This project does not require private API key because open-meteo API is publicly accessible
Python package used so far:
- pandas
- requests
- openmeteo-requests
- requests-cache
- retry-requests
- numpy
- os
- zipfile

# Running analysis 
1. install required packages: `pip install -r requiremenets.txt`
2. From `src/` directory run:`python main.py `
Results will appear in `results/` folder. All obtained will be stored in `data/`