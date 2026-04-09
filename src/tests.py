import pandas as pd
from process import clean_trip_data, clean_station_data, clean_weather_data, calculate_hourly_station_flow, calculate_pressure_index

def test_clean_trip_data():
    sample_trip_data = pd.DataFrame({
        "started_at":["2025-11-01 08:18:00"],
        "ended_at":["2025-11-01 08:48:00"],
        "start_station_id":["A37002"],
        "end_station_id":["A34711"],
    })

    cleaned_data = clean_trip_data(sample_trip_data)

    assert "start_hour" in cleaned_data.columns
    assert "end_hour" in cleaned_data.columns
    print("Clean trip test passed")

def test_clean_station_data():
    sample_station_data = pd.DataFrame({
        "Lat":[42.58],
        "Long":[49.71],
        "Municipality":["Boston"],
        "Total Docks":[20],
        "Number":["A02632"]
    })

    cleaned_data = clean_station_data(sample_station_data)

    assert "station_id" in cleaned_data.columns
    assert "lat" in cleaned_data.columns
    assert "lon" in cleaned_data.columns
    assert "total_docks" in cleaned_data.columns
    print("Clean station test passed")

def test_clean_weather_data():
    sample_weather_data = pd.DataFrame({
        "data":["2025-11-01 08:00:00"],
        "temperature_2m":[2.3]
    })

    cleaned_data = clean_weather_data(sample_weather_data)
    assert "hour" in cleaned_data.columns
    print("Clean weather test passed")

def test_calculate_hourly_station_flow():
    sample_hourly_station_flow = pd.DataFrame({
        "start_station_id": ["A37002", "A37002"],
        "start_hour": ["2025-11-01 08:00:00", "2025-11-01 08:00:00"],
        "end_station_id": ["A34711", "A34711"],
        "end_hour": ["2025-11-01 08:00:00", "2025-11-01 08:00:00"]
    })
    flow_data = calculate_hourly_station_flow(sample_hourly_station_flow)

    assert "arrivals" in flow_data.columns
    assert "departures" in flow_data.columns
    assert "net_flow" in flow_data.columns
    print("Calculate hourly station flow test passed")

def test_calculate_pressure_index():
    sample_flow_data = pd.DataFrame({
        "station_id":["A37002"],
        "hour":["2025-11-01 08:00:00"],
        "net_flow":[3]
    })
    sample_station_data = pd.DataFrame({
        "station_id":["A37002"],
        "total_docks":[20],
    })
    pressure_data = calculate_pressure_index(sample_flow_data, sample_station_data)

    assert "pressure_index" in pressure_data.columns
    print("Calculate pressure index test passed")

if __name__ == "__main__":
    test_clean_trip_data()
    test_clean_station_data()
    test_clean_weather_data()
    test_calculate_hourly_station_flow()
    test_calculate_pressure_index()
    print("All tests passed")