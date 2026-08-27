"""Metrics: overall R2/RMSE/MAE + between/within-station R2 decomposition.

The decomposition answers "does the model predict pollution or just place":
  between-station R2: per-station means of prediction vs truth
  within-station R2: anomalies (value minus own-station mean) — temporal skill

Pure numpy/pandas — no sklearn/scipy (their compiled wheels stopped loading on
Darwin 27; every quantity here is a few lines of numpy anyway).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _r2(y_true, y_pred) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    ss_res = float(((yt - yp) ** 2).sum())
    ss_tot = float(((yt - yt.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, station_ids: np.ndarray) -> dict:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    out = {
        "r2": _r2(yt, yp),
        "rmse": float(np.sqrt(((yt - yp) ** 2).mean())),
        "mae": float(np.abs(yt - yp).mean()),
        "n": int(len(yt)),
    }
    df = pd.DataFrame({"sid": station_ids, "yt": yt, "yp": yp})
    per_station = df.groupby("sid").agg(yt=("yt", "mean"), yp=("yp", "mean"))
    if len(per_station) >= 3:
        out["between_station_r2"] = _r2(per_station["yt"], per_station["yp"])
    means = df.groupby("sid").transform("mean")
    anom_t = df["yt"] - means["yt"]
    anom_p = df["yp"] - means["yp"]
    if anom_t.abs().sum() > 0:
        out["within_station_r2"] = _r2(anom_t, anom_p)
    return out


def roc_auc(y_true, score) -> float:
    """Rank-based AUC (Mann-Whitney), tie-aware; matches sklearn.roc_auc_score."""
    t = np.asarray(y_true, dtype=bool)
    s = np.asarray(score, dtype=float)
    n_pos, n_neg = int(t.sum()), int((~t).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    # midranks for ties
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[t].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def classification_metrics(y_true, y_pred, threshold: float = 35.0) -> dict:
    """Regression outputs -> exceedance-detection metrics.

    Binary task: PM2.5 > threshold (EPA 'Unhealthy for Sensitive Groups'
    boundary, ug/m3). AUC uses the continuous prediction as ranking score;
    accuracy/F1 use the thresholded prediction.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    t = y_true > threshold
    p = y_pred > threshold
    tp = float((t & p).sum())
    f1 = 2 * tp / max(2 * tp + float((~t & p).sum()) + float((t & ~p).sum()), 1.0)
    out = {
        "exceed_prevalence": float(t.mean()),
        "accuracy": float((t == p).mean()),
        "f1": f1,
    }
    out["auc"] = roc_auc(t, y_pred) if 0 < t.sum() < len(t) else float("nan")
    return out
