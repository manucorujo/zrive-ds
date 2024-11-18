import pandas as pd
import os
import logging

from sklearn.metrics import precision_recall_curve, auc
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from joblib import load
from aux_functions import (
    train_val_split,
    remove_orders_few_products,
)

DATA_PATH = Path.cwd().parents[0].joinpath("zrive-data", "feature_frame.csv")
OUTPUT_PATH = Path.cwd().parents[0].joinpath("zrive-data", "output")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
FEATURE_COLS = [
    "ordered_before",
    "abandoned_before",
    "global_popularity",
]
MODEL_NAME = "best_model.pkl"


def main():
    model = load(os.path.join(OUTPUT_PATH, MODEL_NAME))

    data = pd.read_csv(DATA_PATH)
    filtered_data = remove_orders_few_products(data, 5)
    _, _, _, _, x_test, y_test = train_val_split(filtered_data)
    scaler = StandardScaler()
    x_test_scaled = scaler.fit_transform(x_test[FEATURE_COLS])

    y_pred = model.predict_proba(x_test_scaled)[:, 1]

    pr_test, r_test, _ = precision_recall_curve(y_test, y_pred)
    pr_auc = auc(r_test, pr_test)

    logging.info(f"PR AUC: {pr_auc}")
    logging.info(f"Model loaded and evaluated.")


if __name__ == "__main__":
    main()
