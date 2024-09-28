import requests
import pandas as pd

API_URL = "https://archive-api.open-meteo.com/v1/archive"
COORDINATES = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}
VARIABLES = ["temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max"]
START_DATE = "2010-01-01"
END_DATE = "2020-12-31"


def get_data():
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

                # Create a DataFrame from the daily data
                df = pd.DataFrame(
                    {
                        "time": daily_data["time"],
                        "temperature_2m_mean": daily_data["temperature_2m_mean"],
                        "precipitation_sum": daily_data["precipitation_sum"],
                        "wind_speed_10m_max": daily_data["wind_speed_10m_max"],
                    }
                )

                # Convert 'time' column to datetime format
                df["time"] = pd.to_datetime(df["time"])

                print(f"Data for {city}:")
                print(df.head())
            else:
                print(f"No daily data found for {city}.")
        else:
            raise Exception(f"Error{r.status_code}: {r.text}")


def main():
    get_data()


if __name__ == "__main__":
    main()
