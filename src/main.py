from config import TRIP_DATA_URL, DATA_FOLDER, STATION_DATA_URL, OPEN_METEO_URL, BOSTON_LATITUDE, BOSTON_LONGITUDE, START_DATE, END_DATE, HOURLY_VARIABLES
from load_data import download_trip_data, load_trip_data, download_station_data, load_station_data, get_weather_data


def main():
    trip_csv_path = download_trip_data(TRIP_DATA_URL, DATA_FOLDER)
    if trip_csv_path is not None:
        trip_data = load_trip_data(trip_csv_path)
        if trip_data is not None:
            print("Trip Data Volumn:", trip_data.shape)
            print("Trip Data Columns:", trip_data.columns)
        print()

    station_csv_path = download_station_data(STATION_DATA_URL, DATA_FOLDER)
    if station_csv_path is not None:
        station_data = load_station_data(station_csv_path)
        if station_data is not None:
            print("Station Data Volumn:", station_data.shape)
            print("Station Data Columns:", station_data.columns)

    weather_df = get_weather_data(OPEN_METEO_URL, BOSTON_LATITUDE, BOSTON_LONGITUDE, START_DATE, END_DATE, HOURLY_VARIABLES)
    if weather_df is not None:
        print("Weather Data Volumn:", weather_df.shape)
        print("Weather Data Columns:", weather_df.columns.tolist())
        print(weather_df.head())

if __name__ == "__main__":
    main()