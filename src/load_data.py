import os
import zipfile
import requests
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry

# --- 1. DOWNLOAD DATA FROM BLUEBIKE TRIP HISTORY WEBSITE ---
def download_trip_data(url, download_folder):
    """
    Download trip data zip file from a url and extrat it.
    :param url: direct url to the zip file
    :param download_folder: folder where the zip and csv will be saved
    :return: extracted csv file path, or None if the download is failed
    """
    print(f"--- Downloading trip data from {url} ---")
    try:
        os.makedirs(download_folder, exist_ok=True)
        trip_zip_filename = os.path.basename(url)
        trip_zip_path = os.path.join(download_folder, trip_zip_filename)

        response = requests.get(url)
        response.raise_for_status()

        with open(trip_zip_path, "wb") as trip_file:
            trip_file.write(response.content)
        print(f"Trip Zip File Downloaded to: {trip_zip_path}")

        with zipfile.ZipFile(trip_zip_path, "r") as trip_zip_ref:
            trip_zip_ref.extractall(download_folder)
            extracted_files = trip_zip_ref.namelist()
        print("Trip data extracted successful")

        if extracted_files:
            csv_path = os.path.join(download_folder, extracted_files[0])
            print(f"Trip CSV File Downloaded to: {csv_path}")
            return csv_path
        print("No files were extracted from trip zip file")
        return None
    except Exception as e:
        print(f"Error downloading trip data")
        return None

def load_trip_data(file_path):
    """
    Load trip data from downloaded csv file.
    :param file_path: path to the trip data csv file
    :return: trip data dataframe or None if no trip data was downloaded
    """
    print(f"--- Loading trip data from {file_path} ---")
    try:
        df = pd.read_csv(file_path)
        print(f"Trip Data Loaded Successfully: {file_path}")
        return df
    except Exception as e:
        print(f"Error: Could not load trip data from {file_path}")
        return None

# --- 2. DOWNLOAD DATA FROM BLUEBIKE STATION HISTORY WEBSITE ---
def download_station_data(url, download_folder):
    """
    Download station data as a csv file from a url.
    :param url: direct url to the csv file
    :param download_folder:  folder csv will be saved
    :return: downloaded csv fie path, or None if the download is failed
    """
    print(f"--- Downloading station data from {url} ---")
    try:
        os.makedirs(download_folder, exist_ok=True)
        csv_filename = os.path.basename(url)
        csv_path = os.path.join(download_folder, csv_filename)

        response = requests.get(url)
        response.raise_for_status()

        with open(csv_path, "wb") as station_csv_file:
            station_csv_file.write(response.content)
        print(f"Station CSV File Downloaded to: {csv_path}")
        return csv_path
    except Exception as e:
        print(f"Error downloading station data: {e}")
        return None

def load_station_data(file_path):
    """
    Load station data from downloaded csv file.
    :param file_path: path to the station data csv file
    :return: station data dataframe or None if no station data was downloaded
    """
    print(f"--- Loading station data from {file_path} ---")
    try:
        df = pd.read_csv(file_path, skiprows=1)
        print(f"Station Data Loaded Successfully: {file_path}")
        return df
    except Exception as e:
        print(f"Error: Could not load station data from {file_path}")
        return None

# --- 3. API DATA OF WEATHER ---
def get_weather_data(url, latitude, longitude, start_date, end_date, hourly_variables):
    """
    Get hourly weather data from open-meteo.com API using one single boston coordinate
    :param url: open-meteo API url
    :param latitude: latitude of selected location
    :param longitude: longitude of selected location
    :param start_date: start date in YYYY-MM-DD format
    :param end_date: end date in YYYY-MM-DD format
    :param hourly_variables: list of the hourly variables
    :return: pandas DataFrame or None if no data was get from API
    """
    print(f"--- Getting weather data from open-meteo API ---")

    try:
        cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)

        params = {
            "latitude": 42.3501,
            "longitude": -71.0589,
            "start_date": "2025-10-31",
            "end_date": "2025-11-30",
            "hourly": hourly_variables,
        }
        responses = openmeteo.weather_api(url, params=params)

        response = responses[0]
        print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
        print(f"Elevation: {response.Elevation()} m asl")
        print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")

        hourly = response.Hourly()
        hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
        hourly_precipitation = hourly.Variables(1).ValuesAsNumpy()
        hourly_snowfall = hourly.Variables(2).ValuesAsNumpy()
        hourly_wind_speed_10m = hourly.Variables(3).ValuesAsNumpy()
        hourly_cloud_cover = hourly.Variables(4).ValuesAsNumpy()

        hourly_data = {
            "data": pd.date_range(
                start = pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end = pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq = pd.Timedelta(seconds = hourly.Interval()),
                inclusive = "left"
            )
        }

        hourly_data["temperature_2m"] = hourly_temperature_2m
        hourly_data["precipitation"] = hourly_precipitation
        hourly_data["snowfall"] = hourly_snowfall
        hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
        hourly_data["cloud_cover"] = hourly_cloud_cover

        hourly_dataframe = pd.DataFrame(data = hourly_data)
        return hourly_dataframe

    except Exception as e:
        print(f"Error getting weather data: {e}")
        return None