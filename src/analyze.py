from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

# --- Step 1: Analyze how hour of the day influence pressure index ---
def analyze_hourly_imbalance(df, severe_threshold=0.2):
    """
    calculate the hourly imbalance using the absolute pressure index
    :param df: final dataset with hour_of_day and pressure_index
    :param severe_threshold: threshold for severe imbalance
    :return: DataFrame with hourly imbalance
    """
    analysis_df = df.copy()

    #Uses the absolute value to find out the case of severe imbalance regardless of it is inflow or outflow
    analysis_df["abs_pressure_index"] = analysis_df["pressure_index"].abs()

    analysis_df["is_severe"] = analysis_df["abs_pressure_index"] > severe_threshold

    hourly_summary = analysis_df.groupby("hour_of_day").agg(
        mean_abs_pressure=("abs_pressure_index", "mean"),
        severe_percent=("is_severe", "mean"),
    ).reset_index()

    hourly_summary["severe_percent"] = hourly_summary["severe_percent"] * 100

    return hourly_summary

# --- Step 2: Analyze how day of the week influence pressure index ---
def analyze_weekday_imbalance(df, severe_threshold=0.2):
    """
    calculate the weekday imbalance using the absolute pressure index
    :param df: final dataset
    :param severe_threshold: threshold for severe imbalance
    :return: DataFrame with weekday imbalance summary
    """
    analysis_df = df.copy()

    analysis_df["abs_pressure_index"] = analysis_df["pressure_index"].abs()
    analysis_df["is_severe"] = analysis_df["abs_pressure_index"] > severe_threshold

    weekday_summary = analysis_df.groupby("day_of_week").agg(
        mean_abs_pressure=("abs_pressure_index", "mean"),
        severe_percent=("is_severe", "mean"),
    ).reset_index()

    weekday_summary["severe_percent"] = weekday_summary["severe_percent"] * 100

    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_summary["day_of_week"] = pd.Categorical(
        weekday_summary["day_of_week"],
        categories=order,
        ordered=True
    )

    weekday_summary = weekday_summary.sort_values("day_of_week")

    return weekday_summary

# --- Step 3: Analyze how location influence pressure index ---
def analyze_station_imbalance(df, severe_threshold=0.2):
    """
    Analyze imbalance at station level
    """
    df["abs_pressure_index"] = df["pressure_index"].abs()
    df["is_severe"] = df["abs_pressure_index"] > severe_threshold

    result = (
        df.groupby("station_id")
        .agg(
            mean_abs_pressure=("abs_pressure_index", "mean"),
            severe_percent=("is_severe", "mean"),
        )
        .reset_index()
    )

    result["severe_percent"] = result["severe_percent"] * 100
    return result

def prepare_station_map_data(df):
    map_df = df.copy()
    map_df["abs_pressure"] = map_df["pressure_index"].abs()
    station_map = (
        map_df.groupby(["station_id", "lat", "lon", "municipality"])
        .agg(mean_abs_pressure=("abs_pressure_index", "mean"))
        .reset_index()
    )
    return station_map