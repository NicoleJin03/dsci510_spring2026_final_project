import pandas as pd

# --- Step 1: Clean Station Data ---
def clean_station_data(df):
    """
    Clean and minimize only necessary data from the downloaded Bluebikes Station Data
    :param df: raw station DataFrame
    :return: cleaned station data DataFrame
    """
    cleaned_df = df.copy()

    #Rename the useful columns to standardize
    cleaned_df = cleaned_df.rename(columns={
        "Number":"station_id",
        "Lat":"lat",
        "Long":"lon",
        "Municipality":"municipality",
        "Total Docks":"total_docks",
    })

    #Only keep useful columns
    cleaned_df = cleaned_df[
        [
            "station_id", "lat", "lon", "municipality", "total_docks",

        ]
    ]

    #Clean data types
    cleaned_df["total_docks"] = pd.to_numeric(cleaned_df["total_docks"], errors="coerce")
    cleaned_df["lat"] = pd.to_numeric(cleaned_df["lat"], errors="coerce")
    cleaned_df["lon"] = pd.to_numeric(cleaned_df["lon"], errors="coerce")

    return cleaned_df

# --- Step 2: Clean trip data ---
def clean_trip_data(df):
    """
    Clean and prepare the necessary data from the downloaded Bluebikes Trip Data
    :param df: raw trip data DataFrame
    :return: cleaned trip data DataFrame
    """

    cleaned_df = df.copy()

    #Covert time stamps to datetime
    cleaned_df["started_at"] = pd.to_datetime(cleaned_df["started_at"], errors="coerce")
    cleaned_df["ended_at"] = pd.to_datetime(cleaned_df["ended_at"], errors="coerce")

    #Create hourly time stamps
    cleaned_df["start_hour"] = cleaned_df["started_at"].dt.floor("h")
    cleaned_df["end_hour"] = cleaned_df["ended_at"].dt.floor("h")

    #standardized station id type
    cleaned_df["start_station_id"] = cleaned_df["start_station_id"].astype(str)
    cleaned_df["end_station_id"] = cleaned_df["end_station_id"].astype(str)

    return cleaned_df

# --- Step 3: Clean weather data ---
def clean_weather_data(df):
    """
    Clean and prepare the necessary data from the downloaded API Weather Data
    :param df: raw weather data DataFrame
    :return: cleaned weather data DataFrame
    """
    cleaned_df = df.copy()

    #rename time column
    cleaned_df = cleaned_df.rename(columns={"data":"hour"})

    #make hour as datetime
    cleaned_df["hour"] = pd.to_datetime(cleaned_df["hour"],utc=True, errors="coerce").dt.tz_localize(None)

    return cleaned_df

# --- Step 4: Calculate the number of trip flow of each station in each hour ---
def calculate_hourly_station_flow(df):
    """
    calculate the hourly station flow by looking at the arrivals, departures, and net flow by sttaion
    :param df: cleaned trip data DataFrame
    :return: hourly station flow DataFrame
    """

    #count how many trips start from each station in each hour
    departures = df.groupby(["start_station_id","start_hour"]).size().reset_index()
    departures.columns = ["station_id","hour","departures"]

    #count how many trip end at each station in each hour
    arrivals = df.groupby(["end_station_id","end_hour"]).size().reset_index()
    arrivals.columns = ["station_id","hour","arrivals"]

    #merge the arrivals and departures table together
    flow_df = pd.merge(arrivals, departures, on=["station_id","hour"], how="outer")

    #fill 0 if no arrivals or departures in a station for the hour
    flow_df["departures"] = flow_df["departures"].fillna(0)
    flow_df["arrivals"] = flow_df["arrivals"].fillna(0)

    #calculate the net flow of each hour of each station
    # if positive -> more bikes arrived than left
    # if negative -> more bikes left than arrived
    flow_df["net_flow"] = flow_df["arrivals"] - flow_df["departures"]

    return flow_df

# --- Step 5. Calculate pressure index of each dock ---
def calculate_pressure_index(flow_df, station_df):
    """
    Merge hourly station flow with station dock capacity to calculate pressure index
    :param flow_df: hourly station flow DataFrame
    :param station_df: cleaned trip data DataFrame
    :return: DataFrame with pressure index
    """

    #merge hourly flow with station information
    merged_df = pd.merge(flow_df, station_df, on=["station_id"], how="left")

    #calculate pressure index
    merged_df["pressure_index"] = merged_df["net_flow"] / merged_df["total_docks"]

    return merged_df

# --- Step 6. Add weather data to the current data by hour ---
def merge_weather_data(pressure_df, weather_df):
    """
    Merge weather data by hour
    :param pressure_df: pressure index DataFrame
    :param weather_df: cleaned weather DataFrame
    :return: merged final DataFrame
    """
    final_df = pd.merge(pressure_df, weather_df, on="hour", how="left")
    return final_df
