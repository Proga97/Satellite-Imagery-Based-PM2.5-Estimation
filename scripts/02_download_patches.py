#!/usr/bin/env python
"""Download station-week Sentinel-2 patches via GEE computePixels."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from thesis.config import load_config
from thesis.gee.auth import init_ee
from thesis.gee.export import PatchJob, run_downloads


def select_jobs(cfg, limit_stations: int | None, limit_weeks: int | None,
                weeks_per_month: int | None) -> list[PatchJob]:
    stations = pd.read_parquet(cfg.path("stations"))
    weekly = pd.read_parquet(cfg.path("labels_weekly"))

    if limit_stations:
        # spread selection across the latitude range, not just the head of the list
        stations = stations.sort_values("lat")
        step = max(1, len(stations) // limit_stations)
        stations = stations.iloc[::step].head(limit_stations)
    weekly = weekly[weekly["station_id"].isin(stations["station_id"])]

    weekly = weekly.copy()
    weekly["week_start"] = pd.to_datetime(weekly["week_start"])
    if weeks_per_month:
        weekly["ym"] = weekly["week_start"].dt.to_period("M")
        weekly = (weekly.sort_values("week_start")
                  .groupby(["station_id", "ym"]).head(weeks_per_month))
    if limit_weeks:
        keep_weeks = sorted(weekly["week_start"].unique())
        step = max(1, len(keep_weeks) // limit_weeks)
        keep = set(keep_weeks[::step][:limit_weeks])
        weekly = weekly[weekly["week_start"].isin(keep)]

    meta = stations.set_index("station_id")
    jobs = []
    for row in weekly.itertuples():
        lat = float(meta.at[row.station_id, "lat"])
        lon = float(meta.at[row.station_id, "lon"])
        jobs.append(PatchJob(
            station_id=row.station_id,
            week_start=row.week_start.strftime("%Y-%m-%d"),
            lat=lat, lon=lon, epsg=cfg.utm_epsg(lon),
        ))
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-stations", type=int, default=None)
    parser.add_argument("--limit-weeks", type=int, default=None)
    parser.add_argument("--weeks-per-month", type=int, default=None,
                        help="subsample to N weeks per station-month (full run: 2)")
    parser.add_argument("--product", choices=["l2a", "l1c"], default=None,
                        help="override configs/pipeline.yaml patches.product")
    args = parser.parse_args()

    cfg = load_config()
    if args.product:
        cfg.patches["product"] = args.product
    init_ee(cfg.ee_project)

    jobs = select_jobs(cfg, args.limit_stations, args.limit_weeks, args.weeks_per_month)
    print(f"{len(jobs)} station-week jobs "
          f"({len({j.station_id for j in jobs})} stations, product={cfg.patches['product']})")

    manifest = run_downloads(jobs, cfg.patches, cfg.path("patches_dir"), cfg.path("manifest"))
    counts = manifest[manifest["product"] == cfg.patches["product"]]["status"].value_counts()
    print("\nmanifest status counts:")
    print(counts.to_string())
    errs = manifest[(manifest["status"] == "error")]
    if len(errs):
        print("\nsample errors:")
        print(errs["error"].head(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
