"""Train/test split protocols. Correctness here decides result validity."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def random_split(table: pd.DataFrame, test_frac: float = 0.2, seed: int = 0):
    """Row-level random split — the deliberately optimistic/leaky reference."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(table))
    n_test = int(len(table) * test_frac)
    yield idx[n_test:], idx[:n_test]


def spatial_folds(table: pd.DataFrame, n_splits: int = 5):
    """Leave-stations-out: no station appears in both train and test."""
    gkf = GroupKFold(n_splits=n_splits)
    groups = table["station_id"].to_numpy()
    for train_idx, test_idx in gkf.split(table, groups=groups):
        train_st = set(groups[train_idx])
        test_st = set(groups[test_idx])
        assert train_st.isdisjoint(test_st), "station leakage across spatial folds"
        yield train_idx, test_idx


def temporal_split(table: pd.DataFrame, test_year: int):
    """Train on all years before test_year, test on test_year."""
    train_idx = np.flatnonzero(table["year"] < test_year)
    test_idx = np.flatnonzero(table["year"] == test_year)
    yield train_idx, test_idx


SPLITS = {"random": random_split, "spatial": spatial_folds, "temporal": temporal_split}
