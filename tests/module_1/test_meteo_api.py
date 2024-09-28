from src.module_1.module_1_meteo_api import get_data_from_api, process_data
import pytest  # noqa: F401
from typing import Any

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
    r = get_data_from_api(API_URL, params)  # noqa: F841

    assert isinstance(r, dict)

    assert "latitude" in r
    assert "longitude" in r
    assert "daily_units" in r
    assert "daily" in r

    # check structure of 'daily_units'
    assert isinstance(r["daily_units"], dict)
    assert "temperature_2m_mean" in r["daily_units"]

    # check structure of 'daily'
    assert isinstance(r["daily"], dict)
    assert "time" in r["daily"]
    assert "temperature_2m_mean" in r["daily"]

    # check types of values in 'daily'
    assert isinstance(r["daily"]["time"], list)
    assert isinstance(r["daily"]["temperature_2m_mean"], list)


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


def test_process_data_empty() -> None:
    data: dict[Any, Any] = {}
    assert process_data(data, "Madrid").empty
