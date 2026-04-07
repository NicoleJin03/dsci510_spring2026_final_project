#Downloading data of bluebike trip histroy file from the website
TRIP_DATA_URL = "https://s3.amazonaws.com/hubway-data/202511-bluebikes-tripdata.csv.zip"
TRIP_ZIP_FILE = "202511-bluebikes-tripdata.csv.zip"
TRIP_CSV_FILE = "202511-bluebikes-tripdata.csv"

#Downloading data of bluebike stations history from the website
STATION_DATA_URL = "https://assets.ctfassets.net/p6ae3zqfb1e3/5YxN7XJodVcCCYXl2mKdbc/5e0cc33e96dcbcb4524fbd6f679f1c66/-External-_11.17.25_Bluebikes_Station_List.csv"
STATION_CSV_FILE = "-External-_11.17.25_Bluebikes_Station_List.csv"

#Folders & File Path
DATA_FOLDER = "../data"
TRIP_ZIP_PATH = f"{DATA_FOLDER}/{TRIP_ZIP_FILE}"
TRIP_CSV_PATH = f"{DATA_FOLDER}/{TRIP_CSV_FILE}"
STATION_CSV_PATH = f"{DATA_FOLDER}/{STATION_CSV_FILE}"

#Weather API
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
#Use one repersentative Boston coordinate for the current stage bc i haven't figure if i need to use bounding box or eact match
BOSTON_LATITUDE = 42.3501
BOSTON_LONGITUDE = -71.0589
START_DATE = "2025-10-31"
END_DATE = "2025-11-30"
HOURLY_VARIABLES = ["temperature_2m", "precipitation", "snowfall", "wind_speed_10m", "cloud_cover"]