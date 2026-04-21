from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# --- Section 1 - Analyze how hour of the day influence pressure index ---
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

# --- Section 1.1 - Hourly imbalance plot ---
def plot_hourly_imbalance(hourly_imbalance, result_dir = "../results", notebook_plot = False):
    """
    Plot average imbalance by hour of day
    """
    os.makedirs(result_dir, exist_ok=True)

    plt.bar(
        hourly_imbalance["hour_of_day"],
        hourly_imbalance["mean_abs_pressure"],
    )

    avg_value = hourly_imbalance["mean_abs_pressure"].mean()

    plt.axhline(
        y=avg_value,
        linestyle="--",
        label=f"Average = {avg_value:.3f}",
    )

    plt.xlabel("Hour of Day")
    plt.ylabel("Mean |Pressure Index|")
    plt.title("Average Imbalance by Hour of Day")
    plt.xticks(range(24))
    plt.legend()

    if not notebook_plot:
        plt.savefig(f"{result_dir}/hourly_mean_imbalance.png")
        plt.close()
    else:
        plt.show()

# --- Section 1.2 - Hourly Severe Imbalance Plot ---
def plot_hourly_severe(hourly_imbalance, result_dir = "../results", notebook_plot = False):
    """
    plot severe imbalance frequency by hour of day
    """
    os.makedirs(result_dir, exist_ok=True)
    plt.figure(figsize=(8, 4))

    plt.plot(
        hourly_imbalance["hour_of_day"],
        hourly_imbalance["severe_percent"],
        marker="o",
    )

    avg_severe = hourly_imbalance["severe_percent"].mean()

    plt.axhline(
        y=avg_severe,
        linestyle="--",
        label=f"Average = {avg_severe:.3f}%",
    )

    plt.xlabel("Hour of Day")
    plt.ylabel("% Severe Imbalanced (|PI| > 0.2)")
    plt.title("Severe Imbalance Frequency by Hour of Day")

    plt.xticks(range(24))
    plt.legend()

    if not notebook_plot:
        plt.savefig(f"{result_dir}/hourly_severe_imbalance.png")
        plt.close()
    else:
        plt.show()

# --- Section 2 - Analyze how day of the week influence pressure index ---
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

# --- Section 2.1 - Weekday imbalance plot ---
def plot_weekday_imbalance(weekday_imbalance, y_limits = None, result_dir = "../results", notebook_plot = False):
    """
    plot average imbalance by day of week
    """
    os.makedirs(result_dir, exist_ok=True)
    plt.figure(figsize=(8, 4))

    plt.bar(
        weekday_imbalance["day_of_week"],
        weekday_imbalance["mean_abs_pressure"],
    )

    avg_value = weekday_imbalance["mean_abs_pressure"].mean()

    plt.axhline(
        y=avg_value,
        linestyle="--",
        label=f"Average = {avg_value:.3f}"
    )

    plt.xlabel("Day of Week")
    plt.ylabel("Mean |Pressure Index|")
    plt.title("Average Imbalance by Day of Week")
    plt.xticks(rotation=45)

    if y_limits is not None:
        plt.ylim(y_limits)

    plt.legend()

    if not notebook_plot:
        plt.savefig(f"{result_dir}/weekday_mean_imbalance.png")
        plt.close()
    else:
        plt.show()

# --- Section 2.2 - Weekday Severe Plot ---
def plot_weekday_severe(weekday_imbalance, y_limits = None, result_dir = "../results", notebook_plot = False):
    """
    plot severe imbalance frequency by day of week
    """
    os.makedirs(result_dir, exist_ok=True)
    plt.figure(figsize=(8, 4))

    plt.plot(
        weekday_imbalance["day_of_week"],
        weekday_imbalance["severe_percent"],
        marker="o"
    )

    avg_severe = weekday_imbalance["severe_percent"].mean()

    plt.axhline(
        y=avg_severe,
        linestyle="--",
        label=f"Average = {avg_severe:.2f}%"
    )

    plt.xlabel("Day of Week")
    plt.ylabel("% Severely Imbalanced (|PI| > 0.2)")
    plt.title("Severe Imbalance Frequency by Day of Week")
    plt.xticks(rotation=45)

    if y_limits is not None:
        plt.ylim(y_limits)

    plt.legend()

    if not notebook_plot:
        plt.savefig(f"{result_dir}/weekday_severe_imbalance.png")
        plt.close()
    else:
        plt.show()

# --- Step 3: Analyze how location influence pressure index ---
def analyze_station_imbalance(df, severe_threshold=0.2):
    """
    Analyze imbalance at station level
    """

    station_df = df.copy()

    station_df["pressure_index"] = pd.to_numeric(station_df["pressure_index"], errors="coerce")
    station_df["abs_pressure_index"] = station_df["pressure_index"].abs()
    station_df["is_severe"] = station_df["abs_pressure_index"] > severe_threshold

    result = (
        station_df.groupby(["station_id", "lat", "lon", "municipality"])
        .agg(
            mean_abs_pressure=("abs_pressure_index", "mean"),
            severe_percent=("is_severe", "mean"),
        )
        .reset_index()
    )

    result["severe_percent"] = result["severe_percent"] * 100
    return result

# --- Section 3.1.1 - Plot Station Imbalance distribution ---
def plot_station_imbalance_distribution(station_imbalance, result_dir = "../results", notebook_plot = False):
    os.makedirs(result_dir, exist_ok=True)
    plt.figure(figsize=(8, 4))

    avg_pressure = station_imbalance["mean_abs_pressure"].mean()
    above_avg_pressure_count = (station_imbalance["mean_abs_pressure"] > avg_pressure).sum()
    total_stations = len(station_imbalance)

    plt.hist(station_imbalance["mean_abs_pressure"], bins=30)

    plt.axvline(
        x=avg_pressure,
        linestyle="--",
        label=f"Average = {avg_pressure:.3f}",
    )

    plt.text(
        x=avg_pressure,
        y=plt.ylim()[1] * 0.85,
        s=f"{above_avg_pressure_count} stations\n({above_avg_pressure_count/total_stations:.1%}) above avg",
        fontsize=10,
        ha="left"
    )

    plt.xlabel("Mean |Pressure Index|")
    plt.ylabel("Number of Stations")
    plt.title("Distribution of Station Imbalance by Pressure Index")
    plt.legend()

    if not notebook_plot:
        plt.savefig(f"{result_dir}/station_imbalance_distribution.png")
        plt.close()
    else:
        plt.show()

# --- Section 3.1.2 - Mean Pressure Hotspot Map ---
def prepare_station_map_data(df, severe_threshold=0.2):
    """
    prepare station level data for location map plotting
    """
    map_df = df.copy()

    map_df["pressure_index"] = pd.to_numeric(map_df["pressure_index"], errors="coerce")
    map_df["abs_pressure_index"] = map_df["pressure_index"].abs()
    map_df["is_severe"] = map_df["abs_pressure_index"] > severe_threshold

    station_map = (
        map_df.groupby(["station_id", "lat", "lon", "municipality"])
        .agg(
            mean_abs_pressure=("abs_pressure_index", "mean"),
            severe_percent=("is_severe", "mean")
        ).reset_index()
    )

    station_map["severe_percent"] = station_map["severe_percent"] * 100
    return station_map

def plot_station_pressure_map(station_map_df, result_dir = "../results", notebook_plot = False):
    os.makedirs(result_dir, exist_ok=True)
    plt.figure(figsize=(12, 12))

    plot_df = station_map_df.sort_values(by="mean_abs_pressure")
    sizes = plot_df["mean_abs_pressure"] * 300

    scatter = plt.scatter(
        plot_df["lon"],
        plot_df["lat"],
        c=plot_df["mean_abs_pressure"],
        s=sizes,
        alpha=1,
        edgecolor="white",
        linewidth=0.2,
    )

    plt.colorbar(scatter, label="Mean |Pressure Index|")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Station Imbalance Hotspots by Location")

    if not notebook_plot:
        plt.savefig(f"{result_dir}/station_pressure_imbalance_map.png")
        plt.close()
    else:
        plt.show()

# --- Section 3.2.1 -  Severe Imbalance Station Distribution ---
def plot_station_severe_imbalance(station_imbalance, result_dir = "../results", notebook_plot = False):
    os.makedirs(result_dir, exist_ok=True)
    plt.figure(figsize=(8, 4))

    avg_severe = station_imbalance["severe_percent"].mean()
    above_avg_severe_count = (station_imbalance["severe_percent"] > avg_severe).sum()
    total_stations = len(station_imbalance)

    plt.hist(station_imbalance["severe_percent"], bins=30)
    plt.axvline(
        x=avg_severe,
        linestyle="--",
        label=f"Average = {avg_severe:.3f}"
    )

    plt.text(
        x=avg_severe,
        y=plt.ylim()[1] * 0.85,
        s=f"{above_avg_severe_count} stations\n({above_avg_severe_count/total_stations:.1%}) above avg",
        fontsize=10,
        ha="left"
    )

    plt.xlabel("% Severe Imbalanced (|PI| > 0.2)")
    plt.ylabel("Number of Stations")
    plt.title("Distribution of Station Imbalance by Severe Imbalance")
    plt.legend()

    if not notebook_plot:
        plt.savefig(f"{result_dir}/station_severe_imbalance_distribution.png")
        plt.close()
    else:
        plt.show()

# --- Section 3.2.2 - Above Average Severe Imbalance Map ---
def plot_above_average_severe_map(station_imbalance, result_dir = "../results", notebook_plot = False):
    os.makedirs(result_dir, exist_ok=True)
    plt.figure(figsize=(12, 12))

    avg_severe = station_imbalance["severe_percent"].mean()
    above_avg_severe = station_imbalance[station_imbalance["severe_percent"] > avg_severe]

    plt.scatter(
        station_imbalance["lon"],
        station_imbalance["lat"],
        alpha=0.15,
        label="All Stations"
    )

    plt.scatter(
        above_avg_severe["lon"],
        above_avg_severe["lat"],
        s=above_avg_severe["severe_percent"],
        alpha=0.8,
        label="Above Average Station Severe Imbalance"
    )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Stations Above Average Severe Imbalance")
    plt.legend()

    if not notebook_plot:
        plt.savefig(f"{result_dir}/station_imbalance_above_avg_map.png")
        plt.close()
    else:
        plt.show()