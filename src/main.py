from load_data import load_trip_data, load_station_data


def main():
    trip_file = "../data/202511-bluebikes-tripdata.csv"
    station_file = "../data/11.17.25_Bluebikes_Station_Lists.csv"

    # load trip data
    trips_df = load_trip_data(trip_file)

    print("Trip data loaded successfully.")
    print("Trip data shape:", trips_df.shape)
    print("Trip columns:")
    print(trips_df.columns.tolist())

    # load station data
    stations_df = load_station_data(station_file)

    print("\nStation data loaded successfully.")
    print("Station data shape:", stations_df.shape)
    print("Station columns:")
    print(stations_df.columns.tolist())


if __name__ == "__main__":
    main()