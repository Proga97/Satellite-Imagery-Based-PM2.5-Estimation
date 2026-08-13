"""Label aggregation: completeness rules, ISO edges, wildfire values kept."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from thesis.labels.epa import aggregate_weekly, filter_station_coverage


def make_daily(station_id, dates, pm25, obs_percent=100.0):
    return pd.DataFrame({
        "station_id": station_id, "date": pd.to_datetime(dates),
        "pm25": pm25, "obs_percent": obs_percent,
        "lat": 34.0, "lon": -118.0, "site_name": "x",
    })


def test_week_needs_min_days():
    # 4 days in one ISO week -> dropped with min_days_per_week=5
    d = make_daily("a", ["2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"], [10.0] * 4)
    wk = aggregate_weekly(d, min_days_per_week=5)
    assert len(wk) == 0
    wk = aggregate_weekly(d, min_days_per_week=4)
    assert len(wk) == 1 and wk["pm25"].iloc[0] == 10.0


def test_week_start_is_monday():
    d = make_daily("a", [f"2023-01-{dd:02d}" for dd in range(2, 9)], list(range(7)))
    wk = aggregate_weekly(d, min_days_per_week=5)
    # 2023-01-02 is a Monday; 01-08 is Sunday of the same ISO week
    assert wk["week_start"].iloc[0] == pd.Timestamp("2023-01-02")
    assert wk["n_days"].iloc[0] == 7


def test_low_obs_percent_days_excluded():
    dates = [f"2023-01-{dd:02d}" for dd in range(2, 9)]
    d = make_daily("a", dates, [10.0] * 7)
    d.loc[:2, "obs_percent"] = 50.0  # 3 incomplete days -> only 4 valid days
    wk = aggregate_weekly(d, min_observation_percent=75, min_days_per_week=5)
    assert len(wk) == 0


def test_wildfire_spike_kept():
    dates = [f"2023-08-{dd:02d}" for dd in range(7, 14)]
    d = make_daily("a", dates, [12, 15, 250.0, 300.0, 180.0, 20, 14])
    wk = aggregate_weekly(d, min_days_per_week=5)
    assert len(wk) == 1
    assert wk["pm25"].iloc[0] > 100  # spikes averaged in, not dropped


def test_year_boundary_iso_week():
    # Dec 30 2024 is a Monday; its ISO week spans into Jan 2025
    dates = ["2024-12-30", "2024-12-31", "2025-01-01", "2025-01-02", "2025-01-03"]
    d = make_daily("a", dates, [10.0] * 5)
    wk = aggregate_weekly(d, min_days_per_week=5)
    assert len(wk) == 1
    assert wk["week_start"].iloc[0] == pd.Timestamp("2024-12-30")


def test_coverage_filter():
    # station a: 80 weeks of 2023-2024, station b: 10 weeks
    mondays = pd.date_range("2023-01-02", periods=80, freq="W-MON")
    wk = pd.DataFrame({
        "station_id": ["a"] * 80 + ["b"] * 10,
        "week_start": list(mondays) + list(mondays[:10]),
        "pm25": 10.0, "n_days": 7,
    })
    out = filter_station_coverage(wk, [2023, 2024], min_coverage=0.7)
    assert set(out["station_id"]) == {"a"}
