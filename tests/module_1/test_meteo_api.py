from src.module_1.module_1_meteo_api import get_data_from_api, process_data
import pytest  # noqa: F401

API_URL = "https://archive-api.open-meteo.com/v1/archive"
COORDINATES = {"Madrid": {"latitude": 40.416775, "longitude": -3.703790}}
VARIABLES = ["temperature_2m_mean"]
START_DATE = "2010-01-01"
END_DATE = "2010-01-02"


def test_get_data_from_api_success() -> None:
    params = {
        "start_date": START_DATE,
        "end_date": END_DATE,
        "latitude": COORDINATES["Madrid"]["latitude"],
        "longitude": COORDINATES["Madrid"]["longitude"],
        "daily": ",".join(VARIABLES),
    }
    assert get_data_from_api(API_URL, params) is not None


def test_get_data_from_api_failure() -> None:
    with pytest.raises(Exception):
        get_data_from_api(API_URL, {"invalid_param": "value"})


def test_process_data() -> None:
    data = {
        "daily": {
            "time": [START_DATE],
            "temperature_2m_mean": [10.0],
            "precipitation_sum": [0.0],
            "wind_speed_10m_max": [0.0],
        }
    }
    assert process_data(data, "Madrid").shape == (1, 3)
