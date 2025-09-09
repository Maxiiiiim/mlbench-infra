import common

import numpy as np
import pandas as pd
import sys

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error
)


METRICS = {
    "roc_auc_score": roc_auc_score,
    "f1_score": f1_score,
    "accuracy_score": accuracy_score,
    "f1_score_avg_macro": lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro'),
    "root_mean_squared_error": root_mean_squared_error,
    "log_loss": log_loss,
    "mean_squared_error": mean_squared_error,
    "mean_absolute_error": mean_absolute_error,
}

# Grader functions
# ================
def grader_default(pred: pd.DataFrame, val: pd.DataFrame, comp: dict):
    metric = METRICS.get(comp["metric"])
    if metric is None:
        common.report_error(f"grader_default() : internal error : metric not found : {comp['METRIC']}")
        common.graceful_exit(1)

    try:
        score = metric(val, pred)
        return score
    except Exception as e:
        common.report_error(f"Greader execution failed : {sys.exc_info()}")
        return np.nan


def grader_image_classification(pred: pd.DataFrame, val: pd.DataFrame, comp: dict):
    """
    Specialized grader for image classification tasks that handles string labels and different prediction formats.
    """
    metric = METRICS.get(comp["metric"])
    if metric is None:
        common.report_error(f"grader_image_classification() : internal error : metric not found : {comp['metric']}")
        common.graceful_exit(1)

    try:
        # Handle different prediction formats
        if isinstance(pred, list):
            # Convert list to pandas Series
            pred = pd.Series(pred)
        elif isinstance(pred, np.ndarray):
            pred = pd.Series(pred)

        # Handle different validation formats
        if isinstance(val, pd.DataFrame):
            val = val.iloc[:, 0] if val.shape[1] == 1 else val
        elif isinstance(val, list):
            val = pd.Series(val)
        elif isinstance(val, np.ndarray):
            val = pd.Series(val)

        # Ensure both pred and val have the same length
        min_len = min(len(pred), len(val))
        pred = pred.iloc[:min_len] if hasattr(pred, 'iloc') else pred[:min_len]
        val = val.iloc[:min_len] if hasattr(val, 'iloc') else val[:min_len]

        # For image classification, we might need to handle string labels
        # Convert to string if needed for comparison
        if pred.dtype == 'object' and val.dtype == 'object':
            pred = pred.astype(str)
            val = val.astype(str)

        score = metric(val, pred)
        return score
    except Exception as e:
        common.report_error(f"Image classification grader execution failed : {sys.exc_info()}")
        return np.nan


GRADERS = {
    "default": grader_default,
    "image_classification": grader_image_classification
}
