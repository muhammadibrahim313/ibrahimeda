from __future__ import annotations
from typing import Iterable, Tuple
import numpy as np

def classification_basic(y_true: Iterable[int], y_pred: Iterable[int]) -> Tuple[float, float, float, float]:
    """Return accuracy, precision, recall, f1 for binary labels 0 or 1."""
    yt = np.asarray(list(y_true)).astype(int)
    yp = np.asarray(list(y_pred)).astype(int)
    assert set(np.unique(yt)).issubset({0,1}), "y_true must be 0 or 1"
    assert set(np.unique(yp)).issubset({0,1}), "y_pred must be 0 or 1"
    tp = int(((yt == 1) & (yp == 1)).sum())
    tn = int(((yt == 0) & (yp == 0)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    acc = (tp + tn) / max(1, len(yt))
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
    return round(acc,4), round(prec,4), round(rec,4), round(f1,4)

def regression_basic(y_true: Iterable[float], y_pred: Iterable[float]) -> Tuple[float, float, float, float]:
    """Return MAE, MSE, RMSE, R2."""
    yt = np.asarray(list(y_true), dtype=float)
    yp = np.asarray(list(y_pred), dtype=float)
    err = yt - yp
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum((yt - yp)**2))
    ss_tot = float(np.sum((yt - np.mean(yt))**2))
    r2 = 1.0 - (ss_res / ss_tot if ss_tot != 0 else 0.0)
    return round(mae,4), round(mse,4), round(rmse,4), round(r2,4)
