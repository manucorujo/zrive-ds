import pandas as pd
import os
import joblib

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc


from aux_functions import (
    remove_orders_few_products,
    train_val_split,
)

# Consider just the features which we have seen previously that are relevant
FEATURE_COLS = [
    "ordered_before",
    "abandoned_before",
    "global_popularity",
]
TARGET_COL = "outcome"
cs = [1e-8, 1e-6, 1e-4]

DATA_PATH = Path.cwd().parents[0].joinpath("zrive-data", "feature_frame.csv")
OUTPUT_PATH = Path.cwd().parents[0].joinpath("zrive-data", "output")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


def ridge_model_selection(data):
    x_train, y_train, x_val, y_val, _, _ = train_val_split(data)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train[FEATURE_COLS])
    x_val_scaled = scaler.fit_transform(x_val[FEATURE_COLS])

    best_pr_auc = 0

    for c in cs:
        lr = LogisticRegression(penalty="l2", C=c)
        lr.fit(x_train_scaled, y_train)
        train_proba = lr.predict_proba(x_train_scaled)[:, 1]
        pr_train, r_train, _ = precision_recall_curve(y_train, train_proba)
        pr_auc = auc(r_train, pr_train)

        val_proba = lr.predict_proba(x_val_scaled)[:, 1]
        pr_val, r_val, _ = precision_recall_curve(y_val, val_proba)
        pr_auc_val = auc(r_val, pr_val)
        if pr_auc_val > best_pr_auc:
            best_pr_auc = pr_auc_val
            best_c = c

    best_model = LogisticRegression(penalty="l2", C=best_c)
    best_model.fit(x_train_scaled, y_train)

    return best_model


def save_model(model, filename):
    joblib.dump(model, filename)


def main():
    data = pd.read_csv(DATA_PATH)
    filtered_data = remove_orders_few_products(data, 5)
    best_model = ridge_model_selection(filtered_data)
    save_model(best_model, os.path.join(OUTPUT_PATH, "best_model.pkl"))


if __name__ == "__main__":
    main()
