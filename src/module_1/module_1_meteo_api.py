import matplotlib.pyplot as plt
import pandas as pd
import requests


API_URL = "https://archive-api.open-meteo.com/v1/archive"
COORDINATES = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}
VARIABLES = ["temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max"]
START_DATE = "2010-01-01"
END_DATE = "2020-01-01"


def get_data():
    cities_values = {}
    variables_str = ",".join(VARIABLES)
    for city, coords in COORDINATES.items():
        params = {
            "start_date": START_DATE,
            "end_date": END_DATE,
            "latitude": coords["latitude"],
            "longitude": coords["longitude"],
            "daily": variables_str,
        }
        r = requests.get(API_URL, params=params)
        if r.status_code == 200:
            data = r.json()
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

                cities_values[city] = df
            else:
                print(f"No daily data found for {city}.")
        else:
            raise Exception(f"Error{r.status_code}: {r.text}")
    return cities_values


def plot_cities_weather(cities_dfs):

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


def main():
    dfs = get_data()
    plot_cities_weather(dfs)


if __name__ == "__main__":
    main()
