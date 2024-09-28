import matplotlib.pyplot as plt
import pandas as pd
import requests
from typing import Any


API_URL = "https://archive-api.open-meteo.com/v1/archive"
COORDINATES = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}
VARIABLES = ["temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max"]
START_DATE = "2010-01-01"
END_DATE = "2020-01-02"


def get_data_from_api(url: str, params: dict[str, Any]) -> Any:
    r = requests.get(url, params=params)
    if r.status_code == 200:
        try:
            return r.json()
        except ValueError:
            raise Exception("Error: Could not decode JSON response")
    else:
        raise Exception(f"Error{r.status_code}: {r.text}")


def process_data(data: Any, city: str) -> pd.DataFrame:
    if "daily" in data:
        daily_data = data["daily"]

        # create a DataFrame from the daily data
        df = pd.DataFrame(
            {
                "time": daily_data["time"],
                "temperature_2m_mean": daily_data["temperature_2m_mean"],
                "precipitation_sum": daily_data["precipitation_sum"],
                "wind_speed_10m_max": daily_data["wind_speed_10m_max"],
            }
        )

        # convert 'time' column to datetime format
        df["time"] = pd.to_datetime(df["time"])

        # resample time column from daily to monthly frequency
        df.set_index("time", inplace=True)
        df = df.resample("ME").mean()
    else:
        df = pd.DataFrame()
        print(f"No daily data found for {city}.")

    return df


def get_data_meteo_api(city: str) -> dict[str, pd.DataFrame]:
    cities_values = {}
    variables_str = ",".join(VARIABLES)
    params = {
        "start_date": START_DATE,
        "end_date": END_DATE,
        "latitude": COORDINATES[city]["latitude"],
        "longitude": COORDINATES[city]["longitude"],
        "daily": variables_str,
    }
    data = get_data_from_api(API_URL, params)
    cities_values[city] = process_data(data, city)

    return cities_values


def plot_cities_weather(cities_dfs: dict[str, pd.DataFrame]) -> None:
    # one figure with 3 subplots (one for each variable)
    _, axes = plt.subplots(3, 1, sharex=True)

    variables = ["temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max"]
    titles = ["Temperature (°C)", "Precipitation (mm)", "Wind Speed (km/h)"]

    for i, var in enumerate(variables):
        for city_name, df in cities_dfs.items():
            axes[i].plot(df.index, df[var], label=city_name)
        axes[i].set_title(titles[i])
        axes[i].legend()
        axes[i].set_ylabel(titles[i])

    axes[2].set_xlabel("Time")

    plt.show()


def main() -> None:
    city_weather_dfs = {}
    for city, _ in COORDINATES.items():
        city_weather_dfs.update(get_data_meteo_api(city))
    plot_cities_weather(city_weather_dfs)


if __name__ == "__main__":
    main()
