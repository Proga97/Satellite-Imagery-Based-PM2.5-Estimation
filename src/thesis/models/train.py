"""Fit/predict wrappers for the tabular regressors."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_model(name: str, seed: int):
    if name == "lightgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(
            n_estimators=2000, learning_rate=0.05, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8, random_state=seed, verbose=-1,
        )
    if name == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=seed)
    raise ValueError(f"unknown model: {name}")


def fit_predict(model_name: str, seed: int, table: pd.DataFrame, feat_cols: list[str],
                train_idx: np.ndarray, test_idx: np.ndarray) -> np.ndarray:
    X = table[feat_cols].to_numpy(dtype=np.float32)
    y = table["pm25"].to_numpy(dtype=np.float32)
    model = make_model(model_name, seed)
    if model_name == "lightgbm":
        # hold out 10% of TRAIN STATIONS for early stopping (group-aware)
        import lightgbm as lgb
        rng = np.random.default_rng(seed)
        train_stations = table["station_id"].iloc[train_idx].unique()
        val_stations = set(rng.choice(train_stations,
                                      size=max(1, len(train_stations) // 10), replace=False))
        is_val = table["station_id"].iloc[train_idx].isin(val_stations).to_numpy()
        tr, va = train_idx[~is_val], train_idx[is_val]
        model.fit(X[tr], y[tr], eval_set=[(X[va], y[va])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    else:
        model.fit(X[train_idx], y[train_idx])
    return model.predict(X[test_idx])
