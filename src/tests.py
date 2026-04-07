from load_data import load_trip_data

def test_load_trip_data():
    file_path = "../data/202511-bluebikes-tripdata.csv"
    df = load_trip_data(file_path)

    assert df is not None
    assert df.shape[0] > 0
    assert "ride_id" in df.columns
    assert "start_station_name" in df.columns
    print("test passed")

if __name__ == "__main__":
    test_load_trip_data()
    print("All tests passed")