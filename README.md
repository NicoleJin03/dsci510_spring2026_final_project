# The Winter Bottleneck: Analyzing Dock Capacity Pressure in Boston’s Bluebikes System
# Project Overview / Introduction
This project explores the operational bottlenecks of Boston’s Bluebikes system during severe winter weather. 

The primary objective is to understand:
- when and where dock imbalance occurs
- whether weather conditions significantly contribute to system pressure

To achieve this, this project constructs a **Net Flow Pressure Index**, defined as:

Pressure Index  = (Arrivals - Departures) / Dock Capacity. 

By combining:
- November 2025 trip history
- station metadata
- Open-Meteo hourly weather data, 
this project analyzes temporal, spatial, and enviromental drivers of system imbalance.

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

Table Format

| Source | Description | Data Scale | Access Method | Key Variables |
|------|------------|-----------|--------------|--------------|
| Bluebikes Trip Data | November 2025 trip history | 344,842 trips (13 columns) | CSV download | time, station IDs, coordinates |
| Bluebikes Station Data | Station metadata | 597 stations (8 columns) | CSV download | station ID, location, dock capacity |
| Open-Meteo Weather API | Hourly weather data (Nov 2025) | ~744 hourly records | API | temperature, precipitation, snowfall, wind speed, cloud cover |

Use of requests_cache
- This project uses `requests_cache` library when retrieving weather data from Open-Meteo API.
- `requests_cache` stores API responses locally so that repeated requests with the same parameters do not need to be sent again. When the code is run multiple times, the cached response is used instead of making a new API call.
- The main reason i choos was because this was provided on the API website as instruction. After doing my research, I believe this is the best method for the API data retrival because the weather data are historical data, it won't change, so I cache the response locally so future runs are faster and more reliablely.

# Methodology
1. Data Processing
   1. automated download of trip history data
   2. automated download of station data
   3. retrieve hourly historical weather data from open-meteo API
   4. clean and standardized the data of trip, station, and weather
   5. create the hourly station bike flow data
   6. calculated the preliminary station pressure index based on next flow and total doc capacity
   7. create the merged dataset containing station, time, pressure index, and weather information, ready to be used for analysis

2. Pressure Index
Pressure Index = net_flow / total_docks
positive = overflow pressure
negative = shortage pressure

# Analysis
The project examines data from three perspectives: Temporal, Spatial, and Weather Impact on imbalance patterns and distributions.
Weather specifically has an overall and conditional analysis for deep dive. 
### 1. Temporal Analysis
Examines variation in pressure imbalance across:
- hour of day
- day of week

### 2. Spatial Analysis
Identifies station-level imbalance patterns and geographic clustering.

### 3. Weather Impact Analysis
Evaluates relationships between weather variables and pressure imbalance using:
- correlation analysis
- conditional analysis under high-demand conditions

### 4. Modeling
A logistic regression model is used to predict the probability of extreme imbalance 
(|pressure_index| > threshold).

Two models are compared:
- Baseline: time and location features
- Extended: time, location, and weather features


# Results 
Overall, system imbalance is driven more by structural demand patterns than by weather conditions.

- Time and location are the primary drivers of system imbalance, especially during peak commute hours.
- Imbalance is spatially concentrated, with high-pressure stations clustered in central Boston.
- Weather shows weak overall correlation with pressure imbalance.
- Weather effects become more noticeable under high-demand conditions.
- Temperature has the strongest weather-related relationship, with colder conditions associated with higher imbalance.
- Logistic regression results confirm that time and location explain most variation, while weather provides only marginal improvement.

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
   1. For Training pipeline: Data download -> data processing -> model training -> evaluation 
      1. use code in terminal -> python src/main.py --train 
      2. or more specific, use -> python src/main.py --train --subset_type loose_combined 
   2. Evaluation pipeline: Load processed dataset -> run modeling 
      1. use code in terminal -> python src.main.py --evaluation

Results and charts will appear in `results/` folder. All obtained will be stored in `data/`