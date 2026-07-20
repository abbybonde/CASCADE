"""
visualizer/metrics.py — Evaluation metrics for ScaleMAP visualizer.

Provides scalar metrics used for assessing model fit quality.
"""

from __future__ import annotations

import numpy as np


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² (coefficient of determination) following sklearn convention.

    Returns ``1 - SS_res / SS_tot`` as a Python ``float``.

    SS_res = sum((y_true - y_pred) ** 2)
    SS_tot = sum((y_true - mean(y_true)) ** 2)

    When ``SS_tot == 0`` (constant ``y_true``) and ``SS_res == 0`` (perfect
    prediction of a constant), returns ``1.0``.  When ``SS_tot == 0`` but
    ``SS_res > 0``, returns ``-inf`` (convention: arbitrarily bad fit).

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth target values of shape ``(n,)`` or ``(n, k)``.
    y_pred : np.ndarray
        Predicted values, same shape as ``y_true``.

    Returns
    -------
    float
        R² score.  Perfect prediction → ``1.0``.  Predicting the mean →
        ``0.0``.  Worse than predicting the mean → negative.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))

    if ss_tot == 0.0:
        # Constant y_true: perfect match is R²=1, any error is -inf
        return 1.0 if ss_res == 0.0 else float("-inf")

    return 1.0 - ss_res / ss_tot
