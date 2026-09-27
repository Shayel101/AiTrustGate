"""Evaluation utilities for the AI Trust Gate baseline classifier.

Provides reusable functions for computing metrics, formatting a confusion
matrix / classification report, and listing misclassified rows. Used by
src/train.py after fitting the model; not meant to be run standalone.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

POSITIVE_LABEL = "harmful"  # the class of interest for a safety gate


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, Any]:
    """Compute accuracy, precision/recall/F1 (for the harmful class), and the confusion matrix."""
    labels = ["benign", "harmful"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_harmful": precision_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
        "recall_harmful": recall_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
        "f1_harmful": f1_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
        "confusion_matrix": {
            "labels": labels,
            "matrix": cm.tolist(),
        },
        "false_positives": int(fp),  # benign requests incorrectly blocked
        "false_negatives": int(fn),  # harmful requests incorrectly allowed
        "true_positives": int(tp),
        "true_negatives": int(tn),
    }
    return metrics


def build_evaluation_report(y_true: pd.Series, y_pred: np.ndarray) -> str:
    """Build a human-readable text report combining the classification report and confusion matrix."""
    labels = ["benign", "harmful"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)

    lines = [
        "AI Trust Gate - Evaluation Report",
        "=" * 40,
        "",
        "Classification report:",
        report,
        "Confusion matrix (rows = actual, columns = predicted):",
        f"labels: {labels}",
        str(cm),
        "",
    ]
    return "\n".join(lines)


def find_misclassified(df: pd.DataFrame, y_true: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
    """Return the rows where the predicted label does not match the true label."""
    mismatch_mask = y_true.reset_index(drop=True) != pd.Series(y_pred)
    result = df.reset_index(drop=True)[mismatch_mask.values].copy()
    result["predicted_label"] = np.array(y_pred)[mismatch_mask.values]
    return result
