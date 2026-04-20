from config import TRIP_DATA_URL, DATA_FOLDER, STATION_DATA_URL, OPEN_METEO_URL, BOSTON_LATITUDE, BOSTON_LONGITUDE, START_DATE, END_DATE, HOURLY_VARIABLES
from load_data import download_trip_data, load_trip_data, download_station_data, load_station_data, get_weather_data
from process import clean_station_data, clean_trip_data, clean_weather_data, calculate_hourly_station_flow, \
    calculate_hourly_station_flow, calculate_pressure_index, merge_weather_data, add_time_features

def main():
    trip_csv_path = download_trip_data(TRIP_DATA_URL, DATA_FOLDER)
    if trip_csv_path is not None:
        trip_data = load_trip_data(trip_csv_path)
        if trip_data is not None:
            print("Raw Trip Data Volumn:", trip_data.shape)
            print("Raw Trip Data Columns:", trip_data.columns)

            cleaned_trip_data = clean_trip_data(trip_data)

            print("\nCleaned trip data shape:", cleaned_trip_data.shape)
            print("Cleaned Trip Data Columns:")
            print(cleaned_trip_data.columns.tolist())
            print(cleaned_trip_data[["start_station_id","started_at","start_hour","end_station_id","ended_at","end_hour"]].head())

            hourly_flow_df = calculate_hourly_station_flow(cleaned_trip_data)

            print("\nHourly Flow Data Volumn:", hourly_flow_df.shape)
            print("Hourly Flow Data Columns:")
            print(hourly_flow_df.columns.tolist())
            print(hourly_flow_df.head())

    station_csv_path = download_station_data(STATION_DATA_URL, DATA_FOLDER)
    if station_csv_path is not None:
        station_data = load_station_data(station_csv_path)
        if station_data is not None:
            print("Raw station columns:")
            print(station_data.columns.tolist())

            cleaned_station_data = clean_station_data(station_data)

            print("\nCleaned station data shape:", cleaned_station_data.shape)
            print("Cleaned Station Data Columns:")
            print(cleaned_station_data.columns.tolist())
            print(cleaned_station_data.head())

    pressure_df = calculate_pressure_index(hourly_flow_df, cleaned_station_data)

    print("\nPressure Data Volumn:", pressure_df.shape)
    print("Pressure Data Columns:")
    print(pressure_df.columns.tolist())
    print(pressure_df[["station_id","hour","net_flow","total_docks","pressure_index"]].head())

    weather_df = get_weather_data(OPEN_METEO_URL, BOSTON_LATITUDE, BOSTON_LONGITUDE, START_DATE, END_DATE, HOURLY_VARIABLES)
    if weather_df is not None:
        print("Weather Data Volumn:", weather_df.shape)
        print("Weather Data Columns:", weather_df.columns.tolist())
        print(weather_df.head())

        cleaned_weather_df = clean_weather_data(weather_df)

        print("\nCleaned weather data shape:", cleaned_weather_df.shape)
        print("Cleaned Weather Data Columns:")
        print(cleaned_weather_df.columns.tolist())
        print(cleaned_weather_df.head())

    final_df = merge_weather_data(pressure_df, cleaned_weather_df)
    final_df = add_time_features(final_df)
    print("\nFinal Weather Data Volumn:", final_df.shape)
    print("Final Weather Data Columns:")
    print(final_df.columns.tolist())
    print(final_df.head())

    final_df.to_csv("../results/final_data.csv", index=False)
    print("Final dataset saved")

if __name__ == "__main__":
    main()