"""Metrics: overall R2/RMSE/MAE + between/within-station R2 decomposition.

The decomposition answers "does the model predict pollution or just place":
  between-station R2: per-station means of prediction vs truth
  within-station R2: anomalies (value minus own-station mean) — temporal skill
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, station_ids: np.ndarray) -> dict:
    out = {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "n": int(len(y_true)),
    }
    df = pd.DataFrame({"sid": station_ids, "yt": y_true, "yp": y_pred})
    per_station = df.groupby("sid").agg(yt=("yt", "mean"), yp=("yp", "mean"))
    if len(per_station) >= 3:
        out["between_station_r2"] = float(r2_score(per_station["yt"], per_station["yp"]))
    means = df.groupby("sid").transform("mean")
    anom_t = df["yt"] - means["yt"]
    anom_p = df["yp"] - means["yp"]
    if anom_t.abs().sum() > 0:
        out["within_station_r2"] = float(r2_score(anom_t, anom_p))
    return out
