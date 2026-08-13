"""Decomposition sanity on constructed data."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from thesis.models.metrics import compute_metrics


def test_perfect_prediction():
    y = np.array([1.0, 2, 3, 4, 5, 6])
    sids = np.array(["a", "a", "b", "b", "c", "c"])
    m = compute_metrics(y, y.copy(), sids)
    assert m["r2"] == 1.0 and m["rmse"] == 0.0
    assert m["between_station_r2"] == 1.0
    assert m["within_station_r2"] == 1.0


def test_station_mean_only_prediction():
    """Predicting each station's mean: between-station perfect, within ~garbage."""
    rng = np.random.default_rng(0)
    sids = np.repeat([f"s{i}" for i in range(10)], 20)
    station_means = np.repeat(rng.normal(15, 5, 10), 20)
    y = station_means + rng.normal(0, 2, 200)
    pred = station_means.copy()
    m = compute_metrics(y, pred, sids)
    assert m["between_station_r2"] > 0.95
    assert m["within_station_r2"] <= 0.0  # constant per station -> no temporal skill
