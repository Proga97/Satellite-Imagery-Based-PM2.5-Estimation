"""Train/test split protocols. Correctness here decides result validity."""
from __future__ import annotations

import numpy as np
import pandas as pd


def random_split(table: pd.DataFrame, test_frac: float = 0.2, seed: int = 0):
    """Row-level random split — the deliberately optimistic/leaky reference."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(table))
    n_test = int(len(table) * test_frac)
    yield idx[n_test:], idx[:n_test]


def spatial_folds(table: pd.DataFrame, n_splits: int = 5):
    """Leave-stations-out: no station appears in both train and test.

    Same greedy size-balancing as sklearn's GroupKFold (largest group first
    into the currently-lightest fold) — sklearn dropped because its compiled
    deps stopped loading on Darwin 27.
    """
    groups = table["station_id"].to_numpy()
    uniq, counts = np.unique(groups, return_counts=True)
    order = np.argsort(-counts, kind="mergesort")
    fold_of = {}
    fold_sizes = np.zeros(n_splits, dtype=int)
    for gi in order:
        f = int(np.argmin(fold_sizes))
        fold_of[uniq[gi]] = f
        fold_sizes[f] += counts[gi]
    fold_idx = np.array([fold_of[g] for g in groups])
    for f in range(n_splits):
        test_idx = np.flatnonzero(fold_idx == f)
        train_idx = np.flatnonzero(fold_idx != f)
        train_st = set(groups[train_idx])
        test_st = set(groups[test_idx])
        assert train_st.isdisjoint(test_st), "station leakage across spatial folds"
        yield train_idx, test_idx


def temporal_split(table: pd.DataFrame, test_year: int):
    """Train on all years before test_year, test on test_year."""
    train_idx = np.flatnonzero(table["year"] < test_year)
    test_idx = np.flatnonzero(table["year"] == test_year)
    yield train_idx, test_idx


def region_split(table: pd.DataFrame):
    """Leave-one-region-out: train on all other regions, test on the held-out one.

    The strongest generalization test: the model has never seen ANY station,
    landscape, or climate from the test region.
    """
    regions = table["region"].to_numpy()
    for reg in pd.unique(regions):
        test_idx = np.flatnonzero(regions == reg)
        train_idx = np.flatnonzero(regions != reg)
        yield train_idx, test_idx


SPLITS = {"random": random_split, "spatial": spatial_folds,
          "temporal": temporal_split, "region": region_split}
