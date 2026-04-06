import pandas as pd


def load_trip_data(file_path): #Load trip data from a downloaded CSV file
    return pd.read_csv(file_path)

def load_station_data(file_path): #Load station data from a downloaded CSV file
    return pd.read_csv(file_path)