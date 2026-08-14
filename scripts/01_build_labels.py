#!/usr/bin/env python
"""EPA AQS daily files -> stations.parquet + labels_weekly.parquet."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from thesis.config import load_config
from thesis.labels import epa


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-coverage-filter", action="store_true",
                        help="keep all stations regardless of weekly coverage")
    args = parser.parse_args()

    cfg = load_config()
    raw_dir = cfg.path("raw_epa")

    daily_parts = []
    for year in cfg.years:
        zp = epa.download_year(year, raw_dir)
        part = epa.read_year(zp, cfg.region["epa_state_code"])
        print(f"{year}: {len(part):,} station-days, {part['station_id'].nunique()} stations")
        daily_parts.append(part)
    daily = pd.concat(daily_parts, ignore_index=True)

    stations = epa.build_stations(daily)
    weekly = epa.aggregate_weekly(
        daily,
        min_observation_percent=cfg.labels["min_observation_percent"],
        min_days_per_week=cfg.labels["min_days_per_week"],
    )
    if not args.no_coverage_filter:
        cov_years = cfg.labels.get("coverage_years", cfg.years)
        cov = weekly[weekly["week_start"].dt.year.isin(cov_years)] \
            if hasattr(weekly["week_start"], "dt") else weekly
        kept = epa.filter_station_coverage(cov, cov_years, cfg.labels["min_week_coverage"])
        weekly = weekly[weekly["station_id"].isin(kept["station_id"].unique())].reset_index(drop=True)
    stations = stations[stations["station_id"].isin(weekly["station_id"].unique())].reset_index(drop=True)

    stations.to_parquet(cfg.path("stations"), index=False)
    weekly.to_parquet(cfg.path("labels_weekly"), index=False)

    daily_out = daily[
        (daily["obs_percent"] >= cfg.labels["min_observation_percent"])
        & daily["station_id"].isin(stations["station_id"])
    ][["station_id", "date", "pm25"]].reset_index(drop=True)
    daily_out.to_parquet(cfg.path("labels_daily"), index=False)
    print(f"daily labels kept: {len(daily_out):,} station-days")

    print(f"\nstations kept: {len(stations)}")
    print(f"station-weeks: {len(weekly):,}")
    print(f"pm25 weekly mean: median={weekly['pm25'].median():.1f} "
          f"p95={weekly['pm25'].quantile(0.95):.1f} max={weekly['pm25'].max():.1f} ug/m3")
    print(f"wrote {cfg.path('stations')}")
    print(f"wrote {cfg.path('labels_weekly')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
