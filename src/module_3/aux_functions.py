import pandas as pd
import logging

from typing import Any
from sklearn.metrics import auc, precision_recall_curve, roc_curve


FEATURE_COLS = [
    "ordered_before",
    "abandoned_before",
    "global_popularity",
]
TARGET_COL = "outcome"


def plot_curves(
    ax: Any,
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    label: str = None,
    curve_type: str = "both",
):
    if curve_type in ["precision-recall", "both"]:
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        pr_auc = auc(recall, precision)

        ax[0].step(
            recall,
            precision,
            label=f"{label} (AUC={pr_auc:.2f})" if label else f"AUC={pr_auc:.2f}",
        )
        ax[0].set_xlabel("Recall")
        ax[0].set_ylabel("Precision")
        ax[0].set_title("Precision-Recall Curve")

    if curve_type in ["roc", "both"]:
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)

        ax[1].plot(
            fpr,
            tpr,
            label=f"{label} (AUC={roc_auc:.2f})" if label else f"AUC={roc_auc:.2f}",
        )
        ax[1].set_xlabel("False Positive Rate")
        ax[1].set_ylabel("True Positive Rate")
        ax[1].set_title("ROC Curve")


def feature_target_split(df, target):
    x = df.drop(columns=[target])
    y = df[target]
    return x, y


def train_val_split(data):
    daily_orders = data.groupby("order_date").order_id.nunique()
    cum_sum_daily_orders = daily_orders.cumsum() / daily_orders.sum()
    train_val_cut = cum_sum_daily_orders[cum_sum_daily_orders <= 0.7].idxmax()
    val_test_cut = cum_sum_daily_orders[cum_sum_daily_orders <= 0.9].idxmax()

    train_data = data[data.order_date <= train_val_cut]
    val_data = data[
        (data.order_date > train_val_cut) & (data.order_date <= val_test_cut)
    ]
    test_data = data[data.order_date > val_test_cut]

    x_train, y_train = feature_target_split(train_data, TARGET_COL)
    x_val, y_val = feature_target_split(val_data, TARGET_COL)
    x_test, y_test = feature_target_split(test_data, TARGET_COL)

    return x_train, y_train, x_val, y_val, x_test, y_test


def remove_orders_few_products(df: pd.DataFrame, min_products: int) -> pd.DataFrame:
    count_products = df.groupby("order_id").outcome.sum().reset_index()

    filtered_orders = count_products[count_products.outcome >= 5]
    filtered_df = df[df["order_id"].isin(filtered_orders["order_id"])]
    logging.info(f"Removed orders with less than {min_products} products.")

    return filtered_df
