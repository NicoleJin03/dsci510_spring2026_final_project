import argparse
import os
import pandas as pd
from config import TRIP_DATA_URL, DATA_FOLDER, STATION_DATA_URL, OPEN_METEO_URL, BOSTON_LATITUDE, BOSTON_LONGITUDE, START_DATE, END_DATE, HOURLY_VARIABLES,RAW_DATA_FOLDER,RESULTS_FOLDER, FINAL_DATA_PATH
from load_data import download_trip_data, load_trip_data, download_station_data, load_station_data, get_weather_data
from process import clean_station_data, clean_trip_data, clean_weather_data,calculate_hourly_station_flow, calculate_pressure_index, merge_weather_data, add_time_features
from analyze import train_logistic_regression_models, summarize_logistic_model_results, plot_logistic_model_metrics, plot_logistic_coefficients, plot_logistic_confusion_matrix

def create_final_dataset():

    os.makedirs(RAW_DATA_FOLDER, exist_ok=True)
    os.makedirs(os.path.dirname(FINAL_DATA_PATH), exist_ok=True)
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    trip_csv_path = download_trip_data(TRIP_DATA_URL, RAW_DATA_FOLDER)
    if trip_csv_path is not None:
        trip_data = load_trip_data(trip_csv_path)
        if trip_data is not None:
            print("Raw Trip Data Volume:", trip_data.shape)
            print("Raw Trip Data Columns:", trip_data.columns)

            cleaned_trip_data = clean_trip_data(trip_data)

            print("\nCleaned trip data shape:", cleaned_trip_data.shape)
            print("Cleaned Trip Data Columns:")
            print(cleaned_trip_data.columns.tolist())
            print(cleaned_trip_data[["start_station_id","started_at","start_hour","end_station_id","ended_at","end_hour"]].head())

            hourly_flow_df = calculate_hourly_station_flow(cleaned_trip_data)

            print("\nHourly Flow Data Volume:", hourly_flow_df.shape)
            print("Hourly Flow Data Columns:")
            print(hourly_flow_df.columns.tolist())
            print(hourly_flow_df.head())

    station_csv_path = download_station_data(STATION_DATA_URL, RAW_DATA_FOLDER)
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

    print("\nPressure Data Volume:", pressure_df.shape)
    print("Pressure Data Columns:")
    print(pressure_df.columns.tolist())
    print(pressure_df[["station_id","hour","net_flow","total_docks","pressure_index"]].head())

    weather_df = get_weather_data(OPEN_METEO_URL, BOSTON_LATITUDE, BOSTON_LONGITUDE, START_DATE, END_DATE, HOURLY_VARIABLES)
    if weather_df is not None:
        print("Weather Data Volume:", weather_df.shape)
        print("Weather Data Columns:", weather_df.columns.tolist())
        print(weather_df.head())

        cleaned_weather_df = clean_weather_data(weather_df)

        print("\nCleaned weather data shape:", cleaned_weather_df.shape)
        print("Cleaned Weather Data Columns:")
        print(cleaned_weather_df.columns.tolist())
        print(cleaned_weather_df.head())

    final_df = merge_weather_data(pressure_df, cleaned_weather_df)
    final_df = add_time_features(final_df)
    print("\nFinal Weather Data Volume:", final_df.shape)
    print("Final Weather Data Columns:")
    print(final_df.columns.tolist())
    print(final_df.head())

    final_df.to_csv(FINAL_DATA_PATH, index=False)
    print(f"Final dataset saved to {FINAL_DATA_PATH}")

    return final_df

def load_final_dataset():
    final_df = pd.read_csv(FINAL_DATA_PATH)
    print(f"Final dataset loaded from {FINAL_DATA_PATH}")
    return final_df

def run_logistic_pipeline(final_df, subset_type):
    model_results = train_logistic_regression_models(
        final_df,
        severe_threshold=0.2,
        subset_type=subset_type
    )

    summary_df = summarize_logistic_model_results(model_results)
    print("\n Logistic Regression Summary")
    print(summary_df)

    plot_logistic_model_metrics(
        model_results,
        result_dir=RESULTS_FOLDER,
        notebook_plot=False
    )

    plot_logistic_confusion_matrix(
        model_results,
        model_name="weather",
        result_dir=RESULTS_FOLDER,
        notebook_plot=False
    )

    plot_logistic_coefficients(
        model_results,
        model_name="weather",
        result_dir=RESULTS_FOLDER,
        notebook_plot=False
    )

def train_pipeline(subset_type):
    final_df = create_final_dataset()
    run_logistic_pipeline(final_df, subset_type)


def evaluation_pipeline(subset_type):
    final_df = load_final_dataset()
    run_logistic_pipeline(final_df, subset_type)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        action="store_true",
        help="Run training pipeline"
    )

    parser.add_argument(
        "--evaluation",
        action="store_true",
        help="Run evaluation pipeline"
    )

    parser.add_argument(
        "--subset_type",
        type=str,
        default="all_data",
        help="Subset type: all_data, loose_combined, strict_combined"
    )

    args = parser.parse_args()

    if args.train == args.evaluation:
        raise ValueError("Please choose exactly one: --train or --evaluation")

    if args.train:
        train_pipeline(args.subset_type)

    if args.evaluation:
        evaluation_pipeline(args.subset_type)

if __name__ == "__main__":
    main()