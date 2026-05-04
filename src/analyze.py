from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from config import SEVERE_THRESHOLD

# --- Section 1 - Analyze how hour of the day influence pressure index ---
def analyze_hourly_imbalance(df, severe_threshold=SEVERE_THRESHOLD):
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

    plt.figure(figsize=(10, 5))

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
    plt.tight_layout()

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
def analyze_weekday_imbalance(df, severe_threshold=SEVERE_THRESHOLD):
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
def analyze_station_imbalance(df, severe_threshold=SEVERE_THRESHOLD):
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
def prepare_station_map_data(df, severe_threshold=SEVERE_THRESHOLD):
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

# --- Section 4 - Analyze whether weather influence pressure index ---
def prepare_weather_feature(df):
    """
    Create weather condition for future analysis
    """
    weather_df = df.copy()

    weather_df["pressure_index"] = pd.to_numeric(weather_df["pressure_index"], errors="coerce")
    weather_df["abs_pressure_index"] = weather_df["pressure_index"].abs()

    weather_df["temperature_2m"] = pd.to_numeric(weather_df["temperature_2m"], errors="coerce")
    weather_df["precipitation"] = pd.to_numeric(weather_df["precipitation"], errors="coerce")
    weather_df["snowfall"] = pd.to_numeric(weather_df["snowfall"], errors="coerce")
    weather_df["wind_speed_10m"] = pd.to_numeric(weather_df["wind_speed_10m"], errors="coerce")
    weather_df["cloud_cover"] = pd.to_numeric(weather_df["cloud_cover"], errors="coerce")

    weather_df["is_rain"] = weather_df["precipitation"] > 0
    weather_df["is_snow"] = weather_df["snowfall"] > 0
    weather_df["is_clear"] = weather_df["cloud_cover"] < 20

    return weather_df

# --- Section 4.1 - Weather condition comparison ---
def analyze_weather_conditions(df):
    """
    compare the mean imbalance across different weather conditions such as rain, snow, and clear weather
    """
    weather_df = prepare_weather_feature(df)

    rain_summary = (
        weather_df.groupby("is_rain")["abs_pressure_index"].mean()
        .reset_index(name="mean_abs_pressure")
    )

    snow_summary = (
        weather_df.groupby("is_snow")["abs_pressure_index"].mean()
        .reset_index(name="mean_abs_pressure")
    )

    clear_summary = (
        weather_df.groupby("is_clear")["abs_pressure_index"].mean()
        .reset_index(name="mean_abs_pressure")
    )

    return rain_summary, snow_summary, clear_summary

# --- Section 4.2 - Weather Correlation Test ---
def analyze_weather_correlation(df):
    """
    check if there is correlation between weather conditions and abs pressure index
    """
    weather_df = prepare_weather_feature(df)

    corr_df = weather_df[["abs_pressure_index", "temperature_2m", "precipitation", "snowfall", "wind_speed_10m", "cloud_cover"]].corr()

    return corr_df

# --- Section 4.3.1 - Plot of Rain Impact ---
def plot_rain_imbalance(df, result_dir = "../results", notebook_plot = False):
    os.makedirs(result_dir, exist_ok=True)
    weather_df = prepare_weather_feature(df)

    summary = (
        weather_df.groupby("is_rain")["abs_pressure_index"].mean()
        .reset_index(name="mean_abs_pressure")
    )

    plt.figure(figsize=(6,4))
    plt.bar(summary["is_rain"].astype(str), summary["mean_abs_pressure"])

    plt.xlabel("Rain")
    plt.ylabel("Mean |Pressure Index|")
    plt.title("Avg Imbalance: Rain vs. No Rain")

    plt.savefig(f"{result_dir}/weather_rain_imbalance.png")
    if notebook_plot:
        plt.show()
    else:
        plt.close()

# --- Section 4.3.2 - Plot of Snow Impact ---
def plot_snow_imbalance(df, result_dir = "../results", notebook_plot = False):
    os.makedirs(result_dir, exist_ok=True)
    weather_df = prepare_weather_feature(df)
    summary = (
        weather_df.groupby("is_snow")["abs_pressure_index"].mean()
        .reset_index(name="mean_abs_pressure")
    )

    plt.figure(figsize=(6,4))
    plt.bar(summary["is_snow"].astype(str), summary["mean_abs_pressure"])

    plt.xlabel("Snow")
    plt.ylabel("Mean |Pressure Index|")
    plt.title("Avg Imbalance: Snow vs. No Snow")

    plt.savefig(f"{result_dir}/weather_snow_imbalance.png")
    if notebook_plot:
        plt.show()
    else:
        plt.close()

# --- Section 4.3.3 - Plot of Clear Weather Impact ---
def plot_clear_imbalance(df, result_dir = "../results", notebook_plot = False):
    os.makedirs(result_dir, exist_ok=True)
    weather_df = prepare_weather_feature(df)
    summary = (
        weather_df.groupby("is_clear")["abs_pressure_index"].mean()
        .reset_index(name="mean_abs_pressure")
    )

    plt.figure(figsize=(6,4))
    plt.bar(summary["is_clear"].astype(str), summary["mean_abs_pressure"])

    plt.xlabel("Clear")
    plt.ylabel("Mean |Pressure Index|")
    plt.title("Avg Imbalance: Clear vs. No Clear")
    plt.savefig(f"{result_dir}/weather_clear.png")
    if notebook_plot:
        plt.show()
    else:
        plt.close()

# --- Section 4.3.4 - Scatter Plot for Temperature ---
def plot_temperature_imbalance(df, result_dir = "../results", notebook_plot = False):
    os.makedirs(result_dir, exist_ok=True)
    weather_df = prepare_weather_feature(df)

    plt.figure(figsize=(8,4))
    plt.scatter(
        weather_df["temperature_2m"],
        weather_df["abs_pressure_index"],
        alpha=0.2,
    )

    plt.xlabel("Temperature")
    plt.ylabel("|Pressure Index|")
    plt.title("Temperature Vs. Imbalance")

    plt.savefig(f"{result_dir}/weather_temperature_vs_imbalance.png")
    if notebook_plot:
        plt.show()
    else:
        plt.close()

# --- Section 4.3.5 - Scatter Plot for Wind Weather ---
def plot_wind_imbalance(df, result_dir = "../results", notebook_plot = False):
    os.makedirs(result_dir, exist_ok=True)
    weather_df = prepare_weather_feature(df)
    plt.figure(figsize=(8,4))
    plt.scatter(
        weather_df["wind_speed_10m"],
        weather_df["abs_pressure_index"],
        alpha=0.2,
    )

    plt.xlabel("Wind Speed")
    plt.ylabel("|Pressure Index|")
    plt.title("Wind Speed Vs. Imbalance")

    plt.savefig(f"{result_dir}/weather_wind_vs_imbalance.png")
    if notebook_plot:
        plt.show()
    else:
        plt.close()

# --- Section 4.4 - Add-on comparison between the three weather variable and Avg Imbalance
def analyze_weather_gap(df):
    """
    compare the avg imbalance under True and False weather condition and calculate the gaps for each weather variables
    """
    weather_df = prepare_weather_feature(df)
    weather_variables = {
        "Rain":"is_rain",
        "Snow":"is_snow",
        "Clear":"is_clear",
    }

    rows = []

    for label, col in weather_variables.items():
        summary = (
            weather_df.groupby(col)["abs_pressure_index"].mean()
            .reset_index(name="mean_abs_pressure")
        )

        false_series = summary.loc[summary[col] == False, "mean_abs_pressure"]
        true_series = summary.loc[summary[col] == True, "mean_abs_pressure"]

        if len(false_series) > 0 and len(true_series) > 0:
            false_mean = false_series.iloc[0]
            true_mean = true_series.iloc[0]
            gap = true_mean - false_mean
            abs_gap = abs(gap)

            rows.append({
                "weather_variable": label,
                "false_mean": false_mean,
                "true_mean": true_mean,
                "gap": gap,
                "abs_gap": abs_gap
            })

    gap_df = pd.DataFrame(rows)
    return gap_df

# --- Section 4.4.1 - Plot for Weather Gap comparison ---
def plot_weather_gap_comparison(df, result_dir = "../results", notebook_plot = False):
    os.makedirs(result_dir, exist_ok=True)
    gap_df = analyze_weather_gap(df)

    if gap_df.empty:
        print("No Valid Weather Comparisons available for this subset")
        return

    plt.figure(figsize=(8,5))

    x = np.arange(len(gap_df))
    width = 0.35

    false_color = "#8ecae6"
    true_color = "#023047"

    plt.bar(
        x - width /2,
        gap_df["false_mean"],
        width,
        label="False",
        color = false_color
    )

    plt.bar(
        x + width /2,
        gap_df["true_mean"],
        width,
        label="True",
        color = true_color
    )

    plt.xticks(x, gap_df["weather_variable"])
    plt.xlabel("Weather Condition")
    plt.ylabel("Mean |Pressure Index|")
    plt.title("Impact of Weather On Imbalance")
    plt.legend()
    plt.savefig(f"{result_dir}/weather_gap_comparison.png")
    if notebook_plot:
        plt.show()
    else:
        plt.close()

### --- Section 4.2.1 - Correlation Bar Chart ---
def plot_weather_correlation(df, result_dir = "../results", notebook_plot = False):
    os.makedirs(result_dir, exist_ok=True)
    df = df.copy()
    df["abs_pressure_index"] = df["pressure_index"].abs()
    weather_cols = [
        "temperature_2m",
        "precipitation",
        "snowfall",
        "wind_speed_10m",
        "cloud_cover"
    ]

    corr = df[["abs_pressure_index"] + weather_cols].corr()

    corr_values = corr["abs_pressure_index"].drop("abs_pressure_index")

    plt.figure(figsize=(8,4))

    plt.bar(corr_values.index, corr_values.values)

    plt.axhline(0, linestyle="--")
    plt.ylabel("Correlation with |Pressure Index|")
    plt.title("Weather Variables's Correlation with |Pressure Index|")

    plt.xticks(rotation=45)

    plt.savefig(f"{result_dir}/weather_correlation.png")
    if notebook_plot:
        plt.show()
    else:
        plt.close()

# --- Section 5 - Analyze Weather Impact During High Demand ---
# --- Section 5.1 - Create Five High Demand Condition Subsets ---
def get_weather_condition_subsets(df):
    """
    Create five high demand condition subsets to better evaluate the impact of weather with smaller sample size.
    conditions are:
    1. Peak hour
    2. peak day
    3. High pressure station
    4. loose combined  = peak hour x high pressure station
    5. strict combined = peak hour x peak day x high pressure station
    """
    working_df = prepare_weather_feature(df)

    # 1. Peak hour
    peak_hour_df = working_df[
        working_df["hour_of_day"].isin([7,8,9,16,17,18])
    ]

    # 2. Peak Day
    peak_day_df = working_df[
        working_df["day_of_week"].isin(["Tuesday", "Wednesday", "Thursday"])
    ]

    # 3. high pressure station
    station_imbalance = analyze_station_imbalance(working_df, severe_threshold=SEVERE_THRESHOLD)
    avg_pressure = station_imbalance["mean_abs_pressure"].mean()

    high_pressure_station_ids = station_imbalance.loc[
        station_imbalance["mean_abs_pressure"] > avg_pressure,
        "station_id"
    ]

    high_pressure_station_df = working_df[
        working_df["station_id"].isin(high_pressure_station_ids)
    ]

    # 4. loose combined  = peak hour x high pressure station
    loose_combined_df = working_df[
        (working_df["hour_of_day"].isin([7,8,9,16,17,18])) &
        (working_df["station_id"].isin(high_pressure_station_ids))
    ]

    # 5. strict combined = peak hour x peak day x high pressure station
    strict_combined_df = working_df[
        (working_df["hour_of_day"].isin([7, 8, 9, 16, 17, 18])) &
        (working_df["day_of_week"].isin(["Tuesday", "Wednesday", "Thursday"])) &
        (working_df["station_id"].isin(high_pressure_station_ids))
    ]

    subsets = {
        "peak_hours": peak_hour_df,
        "peak_day": peak_day_df,
        "high_pressure_station": high_pressure_station_df,
        "loose_combined": loose_combined_df,
        "strict_combined": strict_combined_df
    }
    return subsets

# --- Section 5.2 - Compare avg imbalance under True/False Weather Conditions Under Five High Demand Conditions ---
def analyze_weather_under_conditions(df):

    subsets = get_weather_condition_subsets(df)

    results = {}

    for key, subset in subsets.items():
        gap_df = analyze_weather_gap(subset)
        results[key] = gap_df

    return results

# --- Section 5.3 - Correlation between abs Pressure Index and Five Weather Variable Under Five High Demand Conditions ---
def analyze_weather_correlations_by_conditions(df):
    subsets = get_weather_condition_subsets(df)

    weather_cols = [
        "temperature_2m",
        "precipitation",
        "snowfall",
        "wind_speed_10m",
        "cloud_cover"
    ]

    rows = []

    for condition, subset in subsets.items():
        corr_df = subset[["abs_pressure_index"]+weather_cols].corr()

        for weather_col in weather_cols:
            corr_value = corr_df.loc["abs_pressure_index", weather_col]

            rows.append({
                "condition": condition,
                "weather_col": weather_col,
                'correlation': corr_value,
            })
    results_df = pd.DataFrame(rows)
    return results_df

# --- Section 5.4 - Count number of True/False Data Point for Each Weather Condition Under Ecah Five High Demand Conditions ---
def analyze_weather_sample_sizes(df):
    subsets = get_weather_condition_subsets(df)

    weather_variables = {
        "Rain":"is_rain",
        "Snow":"is_snow",
        "Clear":"is_clear"
    }

    row = []

    for condition, subset in subsets.items():
        for label,col in weather_variables.items():
            false_count = (subset[col] == False).sum()
            true_count = (subset[col] == True).sum()
            total_count = false_count + true_count

            row.append({
                "condition": condition,
                "weather_variable": label,
                "false_count": false_count,
                "true_count": true_count,
                "total_count": total_count,
            })

    sample_size_df = pd.DataFrame(row)
    return sample_size_df

# --- Section 5.2.1 - Plot for Weather impact Comparison ---
def plot_weather_condition_comparison(df, result_dir="../results", notebook_plot=False):
    os.makedirs(result_dir, exist_ok=True)

    results = analyze_weather_under_conditions(df)

    plot_row = []

    for condition, gap_df in results.items():
        for _, row in gap_df.iterrows():
            plot_row.append({
                "condition": condition,
                "weather_variable": row['weather_variable'],
                "abs_gap": row['abs_gap'],
            })

    plot_df = pd.DataFrame(plot_row)

    if plot_df.empty:
        print("No weather conditions were detected")
        return

    weather_order = ["Rain", "Snow", "Clear"]
    condition_order = ["peak_hours", "peak_day", "high_pressure_station", "loose_combined","strict_combined"]

    pivot_df = plot_df.pivot(
        index ="weather_variable",
        columns="condition",
        values="abs_gap"
    )

    pivot_df = pivot_df.reindex(weather_order)
    pivot_df = pivot_df[
        [col for col in condition_order if col in pivot_df.columns]
    ]

    plt.figure(figsize=(12,5))

    x = np.arange(len(pivot_df.index))
    width = 0.15

    for i, condition in enumerate(pivot_df.columns):
        values = pivot_df[condition].fillna(0)

        plt.bar(
            x + (i-2) * width,
            values,
            width = width,
            label = condition,
        )

    plt.xticks(x, pivot_df.index)
    plt.ylabel("Absolute Impact on |Pressure Index|")
    plt.xlabel("Weather Condition")
    plt.title("Weather Impact Under Different High Demand Conditions")
    plt.legend()

    plt.savefig(f"{result_dir}/weather_impact_comparison.png")
    if notebook_plot:
        plt.show()
    else:
        plt.close()

# --- Section 5.3.1 - Plot for Correlation Comparison by Condition ---
def plot_weather_correlation_by_condition(df, result_dir="../results", notebook_plot=False):
    os.makedirs(result_dir, exist_ok=True)
    corr_df = analyze_weather_correlations_by_conditions(df)

    variables = ["temperature_2m", "precipitation", "snowfall", "wind_speed_10m", "cloud_cover"]
    conditions = ["peak_hours", "peak_day", "high_pressure_station", "loose_combined","strict_combined"]

    plt.figure(figsize=(12,5))

    x = np.arange(len(variables))
    width = 0.15

    for i, condition in enumerate(conditions):
        subset_df = corr_df[corr_df["condition"] == condition]
        values = []

        for var in variables:
            match = subset_df.loc[subset_df["weather_col"] == var, "correlation"]
            if len(match) > 0:
                values.append(match.iloc[0])
            else:
                values.append(0)

        plt.bar(
            x + (i - 2)* width,
            values,
            width = width,
            label = condition,
        )

    plt.axhline(y=0, linestyle="--")
    plt.xticks(x, variables, rotation=45)
    plt.ylabel("Correlation with |Pressure Index|")
    plt.xlabel("Weather Variables")
    plt.title("Weather Correlation Under Different High Demand Conditions")
    plt.legend()

    plt.savefig(f"{result_dir}/weather_correlation_by_condition.png")
    if notebook_plot:
        plt.show()
    else:
        plt.close()

# --- Section 5.4.1 - Sample Size Comparison by Condition ---
def plot_weather_sample_sizes(df, result_dir="../results", notebook_plot=False):
    os.makedirs(result_dir, exist_ok=True)
    sample_df = analyze_weather_sample_sizes(df)

    weather_order = ["Rain", "Snow", "Clear"]
    condition_order =  ["peak_hours", "peak_day", "high_pressure_station", "loose_combined","strict_combined"]

    pivot_df = sample_df.pivot(
        index ="weather_variable",
        columns="condition",
        values="true_count"
    )

    pivot_df = pivot_df.reindex(weather_order)
    pivot_df = pivot_df[
        [col for col in condition_order if col in pivot_df.columns]
    ]

    plt.figure(figsize=(12,5))

    x = np.arange(len(pivot_df.index))
    width = 0.15

    for i, condition in enumerate(pivot_df.columns):
        values = pivot_df[condition].fillna(0)

        plt.bar(
            x + (i-2) * width,
            values,
            width = width,
            label = condition,
        )

    plt.xticks(x, pivot_df.index)
    plt.ylabel("True Observation Count")
    plt.xlabel("Weather Conditions")
    plt.title("Weather Sample Size Under Different High Demand Conditions")
    plt.legend()

    plt.savefig(f"{result_dir}/weather_sample_sizes.png")
    if notebook_plot:
        plt.show()
    else:
        plt.close()

# --- Section 6. Logistic Regression Modeling on Predicting extreme pressure events and if weather adds value ---
def prepare_logistic_model_data(df, severe_threshold=SEVERE_THRESHOLD, subset_type="all_data"):
    """
    prepare modeling dataset for logistic regression
    target -> is_extreme_pressure = 1 if abs_pressure_index > severe_threshold, else 0
    """

    model_df = prepare_weather_feature(df).copy()

    model_df["is_extreme_pressure"] = model_df["abs_pressure_index"] > severe_threshold
    model_df["is_extreme_pressure"] = model_df["is_extreme_pressure"].astype(int)

    station_imbalance = analyze_station_imbalance(model_df, severe_threshold=severe_threshold)
    avg_pressure = station_imbalance["mean_abs_pressure"].mean()

    high_pressure_station_ids = station_imbalance.loc[
        station_imbalance["mean_abs_pressure"] > avg_pressure,
        "station_id"
    ]

    model_df["is_high_pressure_station"] = model_df["station_id"].isin(high_pressure_station_ids).astype(int)

    if subset_type == "loose_combined":
        model_df = model_df[
            (model_df["hour_of_day"].isin([7,8,9,16,17,18])) &
            (model_df["is_high_pressure_station"] == 1)
        ].copy()
    elif subset_type == "strict_combined":
        model_df = model_df[
            (model_df["hour_of_day"].isin([7, 8, 9, 16, 17, 18])) &
            (model_df["day_of_week"].isin(["Tuesday","Wednesday","Thursday"])) &
            (model_df["is_high_pressure_station"] == 1)
        ].copy()

    model_df = pd.get_dummies(
        model_df,
        columns=["day_of_week"],
        drop_first = True
    )

    return model_df

def build_logistic_feature_set(model_df):
    """
    build baseline and weather enhanced feature sets
    """
    weekday_cols = [col for col in model_df.columns if col.startswith("day_of_week")]

    baseline_features = [
        "hour_of_day",
        "is_high_pressure_station",
    ] + weekday_cols

    weather_features = baseline_features + [
        "temperature_2m",
        "precipitation",
        "snowfall",
        "wind_speed_10m",
        "cloud_cover",
    ]

    X_baseline = model_df[baseline_features].copy()
    X_weather = model_df[weather_features].copy()
    y = model_df["is_extreme_pressure"].copy()

    return X_baseline, X_weather, y, baseline_features, weather_features

def train_logistic_regression_models(df, severe_threshold=SEVERE_THRESHOLD, subset_type="all_data", test_size=0.2, random_state=42):
    """
    train two logistic regression models
    1. baseline model = time + location
    2. weather model = time + location + weather
    """

    model_df = prepare_logistic_model_data(df, severe_threshold=severe_threshold, subset_type=subset_type)

    X_baseline, X_weather, y, baseline_features, weather_features = build_logistic_feature_set(model_df)

    train_idx, test_idx = train_test_split(
        model_df.index,
        train_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    X_train_baseline = X_baseline.loc[train_idx]
    X_test_baseline = X_baseline.loc[test_idx]

    X_train_weather = X_weather.loc[train_idx]
    X_test_weather = X_weather.loc[test_idx]

    y_train = y.loc[train_idx]
    y_test = y.loc[test_idx]

    scaler_baseline = StandardScaler()
    scaler_weather = StandardScaler()

    X_train_baseline_scaled = scaler_baseline.fit_transform(X_train_baseline)
    X_test_baseline_scaled = scaler_baseline.transform(X_test_baseline)

    X_train_weather_scaled = scaler_weather.fit_transform(X_train_weather)
    X_test_weather_scaled = scaler_weather.transform(X_test_weather)

    baseline_model = LogisticRegression(max_iter = 1000, class_weight = "balanced")
    weather_model = LogisticRegression(max_iter = 1000, class_weight = "balanced")

    baseline_model.fit(X_train_baseline_scaled, y_train)
    weather_model.fit(X_train_weather_scaled, y_train)

    y_pred_baseline = baseline_model.predict(X_test_baseline_scaled)
    y_prob_baseline = baseline_model.predict_proba(X_test_baseline_scaled)[:, 1]

    y_pred_weather = weather_model.predict(X_test_weather_scaled)
    y_prob_weather = weather_model.predict_proba(X_test_weather_scaled)[:, 1]

    results = {
        "subset_type": subset_type,
        "model_df": model_df,
        "y_test": y_test,
        "baseline":{
            "model": baseline_model,
            "scaler": scaler_baseline,
            "X_train": X_train_baseline,
            "X_test": X_test_baseline,
            "y_pred": y_pred_baseline,
            "y_prob": y_prob_baseline,
            "features": baseline_features,
        },
        "weather":{
            "model": weather_model,
            "scaler": scaler_weather,
            "X_train": X_train_weather,
            "X_test": X_test_weather,
            "y_pred": y_pred_weather,
            "y_prob": y_prob_weather,
            "features": weather_features,
        }
    }

    return results

def summarize_logistic_model_results(model_results):
    """
    create a summary table of model performance results
    """

    y_test = model_results["y_test"]

    rows = []

    for model_name in ["baseline", "weather"]:
        y_pred = model_results[model_name]["y_pred"]

        rows.append({
            "model": model_name,
            "accuracy":accuracy_score(y_test, y_pred),
            "precision":precision_score(y_test, y_pred, zero_division=0),
            "recall":recall_score(y_test, y_pred, zero_division=0),
            "f1_score":f1_score(y_test, y_pred, zero_division=0),
        })

    summary_df = pd.DataFrame(rows)
    return summary_df

def extract_logistic_coefficients(model_results, model_name="weather"):
    model = model_results[model_name]["model"]
    feature_names = model_results[model_name]["features"]

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": model.coef_[0],
    })

    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("abs_coefficient", ascending=False)

    return coef_df

# --- Section 6.1 - Model Metric Comparison Plot ---
def plot_logistic_model_metrics(model_results, result_dir="../results", notebook_plot=False):
    """
    Plot accuracy, precison, recall, and F1 score for baseline vs. weather model
    """
    os.makedirs(result_dir, exist_ok=True)
    summary_df = summarize_logistic_model_results(model_results)

    metrics = ["accuracy", "precision", "recall", "f1_score"]
    models = summary_df["model"].tolist()

    plt.figure(figsize=(10,5))

    x = np.arange(len(metrics))
    width = 0.35

    baseline_values = summary_df.loc[summary_df["model"] == "baseline", metrics].iloc[0].tolist()
    weather_values = summary_df.loc[summary_df["model"] == "weather", metrics].iloc[0].tolist()

    plt.bar(x-width/2, baseline_values, width=width, label="Baseline")
    plt.bar(x+width/2, weather_values, width=width, label="Weather")

    plt.xticks(x, metrics)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.xlabel("Metric")
    plt.title(f"Logistic Regression Performance ({model_results['subset_type']})")
    plt.legend()

    plt.savefig(f"{result_dir}/logistic_model_metrics_{model_results['subset_type']}.png")
    if notebook_plot:
        plt.show()
    else:
        plt.close()

# --- Section 6.2 - Confusion Matric Plot ---
def plot_logistic_confusion_matrix(model_results, model_name="weather", result_dir="../results", notebook_plot=False):
    """
    plot confusion matrix for selected logistic regression model.
    """
    os.makedirs(result_dir, exist_ok=True)

    y_test = model_results["y_test"]
    y_pred = model_results[model_name]["y_pred"]

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix: {model_name} ({model_results['subset_type']})")

    plt.savefig(f"{result_dir}/logistic_confusion_matrix_{model_name}_{model_results['subset_type']}.png")
    if notebook_plot:
        plt.show()
    else:
        plt.close()

# --- Section 6.3 - Coefficient Comparison ---
def plot_logistic_coefficients(model_results, model_name="weather", top_n=10, result_dir="../results", notebook_plot=False):
    """
    plot top logistic regression coefficients by absolute size.
    """
    os.makedirs(result_dir, exist_ok=True)

    coef_df = extract_logistic_coefficients(model_results, model_name=model_name).head(top_n)
    coef_df = coef_df.sort_values("coefficient")

    plt.figure(figsize=(8, 5))

    plt.barh(
        coef_df["feature"],
        coef_df["coefficient"]
    )

    plt.axvline(x=0, linestyle="--")
    plt.xlabel("Coefficient")
    plt.ylabel("Feature")
    plt.title(f"Top Logistic Regression Coefficients: {model_name} ({model_results['subset_type']})")

    plt.savefig(f"{result_dir}/logistic_coefficients_{model_name}_{model_results['subset_type']}.png")
    if notebook_plot:
        plt.show()
    else:
        plt.close()