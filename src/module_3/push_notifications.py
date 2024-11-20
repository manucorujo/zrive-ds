import pandas as pd
from typing import Any
import logging

CSV_PATH = "/home/manucorujo/zrive-data/feature_frame.csv"
MIN_PRODUCTS = 5


def remove_orders_few_products(df: pd.DataFrame, min_products: int) -> pd.DataFrame:
    # Sum outcome (A product is in the order if its outcome is 1)
    count_products = df.groupby("order_id").outcome.sum().reset_index()

    filtered_orders = count_products[count_products.outcome >= 5]
    filtered_df = df[df["order_id"].isin(filtered_orders["order_id"])]
    logging.info(f"Removed orders with less than {min_products} products.")

    return filtered_df


def read_data(CSV_PATH: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(CSV_PATH)
        logging.info(f"Data read from {CSV_PATH}.")
        return df
    except FileNotFoundError:
        logging.error(f"File {CSV_PATH} not found.")
        return pd.DataFrame()


def main() -> None:
    df: pd.DataFrame = read_data(CSV_PATH)
    filtered_df: pd.DataFrame = remove_orders_few_products(df, MIN_PRODUCTS)


if __name__ == "__main__":
    main()
