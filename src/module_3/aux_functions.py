from typing import Any
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve


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
