from __future__ import annotations

from typing import Iterable
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import spearmanr, pearsonr


def five_param_func(x: np.ndarray, b1: float, b2: float, b3: float, b4: float, b5: float) -> np.ndarray:
    logistic = 0.5 - 1.0 / (1.0 + np.exp(b2 * (x - b3)))
    return b1 * logistic + b4 * x + b5


def fit_five_param_mapping(pred: np.ndarray, mos: np.ndarray) -> np.ndarray:
    pred = pred.astype(np.float64)
    mos = mos.astype(np.float64)
    try:
        beta0 = [np.max(mos), np.min(mos), np.mean(pred), 0.1, np.mean(mos)]
        params, _ = curve_fit(five_param_func, pred, mos, p0=beta0, maxfev=20000)
        return five_param_func(pred, *params)
    except Exception:
        return pred


def compute_metrics(pred: Iterable[float], mos: Iterable[float], fit_nonlinear_mapping: bool = True) -> dict[str, float]:
    pred_arr = np.asarray(list(pred), dtype=np.float64)
    mos_arr = np.asarray(list(mos), dtype=np.float64)
    assert len(pred_arr) == len(mos_arr) and len(pred_arr) > 1
    mapped = fit_five_param_mapping(pred_arr, mos_arr) if fit_nonlinear_mapping else pred_arr
    plcc = pearsonr(mapped, mos_arr).statistic
    srcc = spearmanr(pred_arr, mos_arr).statistic
    rmse = float(np.sqrt(np.mean((mapped - mos_arr) ** 2)))
    return {'PLCC': float(plcc), 'SRCC': float(srcc), 'RMSE': rmse}
