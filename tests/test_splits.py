"""Leakage guarantees + the station-ID canary."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from thesis.models.splits import random_split, spatial_folds, temporal_split


def make_table(n_stations=20, weeks=30, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    station_effect = rng.normal(15, 6, n_stations)
    for s in range(n_stations):
        for w in range(weeks):
            rows.append({
                "station_id": f"st{s:03d}",
                "year": 2023 if w < weeks // 2 else 2024,
                "pm25": station_effect[s] + rng.normal(0, 1),
                "station_code_feature": float(s),  # memorizable station identity
            })
    return pd.DataFrame(rows)


def test_spatial_folds_disjoint_stations():
    t = make_table()
    for tr, te in spatial_folds(t, n_splits=5):
        assert set(t["station_id"].iloc[tr]).isdisjoint(set(t["station_id"].iloc[te]))


def test_spatial_folds_cover_all_rows():
    t = make_table()
    seen = np.zeros(len(t), dtype=int)
    for _, te in spatial_folds(t, n_splits=5):
        seen[te] += 1
    assert (seen == 1).all()


def test_temporal_split_no_future_in_train():
    t = make_table()
    (tr, te), = list(temporal_split(t, test_year=2024))
    assert (t["year"].iloc[tr] < 2024).all()
    assert (t["year"].iloc[te] == 2024).all()


def test_random_split_sizes():
    t = make_table()
    (tr, te), = list(random_split(t, test_frac=0.2, seed=0))
    assert len(tr) + len(te) == len(t)
    assert abs(len(te) - 0.2 * len(t)) < 2


def test_canary_station_id_feature():
    """A model that memorizes station identity must look great under random
    split and useless under spatial CV. If this fails, splits are leaking."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score

    t = make_table()
    X = t[["station_code_feature"]].to_numpy()
    y = t["pm25"].to_numpy()

    (tr, te), = list(random_split(t, seed=0))
    m = RandomForestRegressor(n_estimators=50, random_state=0).fit(X[tr], y[tr])
    r2_random = r2_score(y[te], m.predict(X[te]))

    r2_spatial = []
    for tr, te in spatial_folds(t, n_splits=5):
        m = RandomForestRegressor(n_estimators=50, random_state=0).fit(X[tr], y[tr])
        r2_spatial.append(r2_score(y[te], m.predict(X[te])))

    assert r2_random > 0.8, f"random-split memorization should score high, got {r2_random:.2f}"
    assert np.mean(r2_spatial) < 0.2, f"spatial CV must kill memorization, got {np.mean(r2_spatial):.2f}"
