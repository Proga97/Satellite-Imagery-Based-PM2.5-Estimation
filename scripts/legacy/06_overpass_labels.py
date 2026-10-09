#!/usr/bin/env python
"""Sync labels to Sentinel-2 overpass days.

1. Query GEE once per station for all S2 acquisition timestamps 2023-2024
   (pass times are identical for L1C and L2A - same acquisitions).
2. Convert to local calendar dates, cache to s2_pass_dates.parquet.
3. Label each station-week by the mean EPA PM2.5 over overpass days only.
"""
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from tqdm import tqdm

from thesis.config import load_config
from thesis.gee.auth import init_ee
from thesis.labels.epa import build_overpass_labels

LOCAL_TZ = "America/Los_Angeles"


def query_pass_dates(cfg, stations: pd.DataFrame) -> pd.DataFrame:
    import ee

    start = f"{min(cfg.years)}-01-01"
    end = f"{max(cfg.years) + 1}-01-01"
    collection = cfg.patches["collections"]["l2a"]

    def one(row):
        point = ee.Geometry.Point([row.lon, row.lat])
        ts = (
            ee.ImageCollection(collection)
            .filterBounds(point)
            .filterDate(start, end)
            .aggregate_array("system:time_start")
            .getInfo()
        )
        dates = (
            pd.to_datetime(pd.Series(ts, dtype="int64"), unit="ms", utc=True)
            .dt.tz_convert(LOCAL_TZ)
            .dt.normalize()
            .dt.tz_localize(None)
            .drop_duplicates()
        )
        return pd.DataFrame({"station_id": row.station_id, "date": dates})

    frames = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(one, row) for row in stations.itertuples()]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="pass dates"):
            frames.append(fut.result())
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-query pass dates")
    args = parser.parse_args()

    cfg = load_config()
    cache = cfg.path("s2_pass_dates")

    if cache.exists() and not args.force:
        pass_dates = pd.read_parquet(cache)
        print(f"cached pass dates: {len(pass_dates):,} station-days")
    else:
        init_ee(cfg.ee_project)
        stations = pd.read_parquet(cfg.path("stations"))
        pass_dates = query_pass_dates(cfg, stations)
        pass_dates.to_parquet(cache, index=False)
        print(f"queried pass dates: {len(pass_dates):,} station-days "
              f"({pass_dates['station_id'].nunique()} stations)")

    daily = pd.read_parquet(cfg.path("labels_daily"))
    overpass = build_overpass_labels(daily, pass_dates)
    out = cfg.path("labels_overpass")
    overpass.to_parquet(out, index=False)

    weekly = pd.read_parquet(cfg.path("labels_weekly"))
    merged = weekly.merge(overpass, on=["station_id", "week_start"],
                          suffixes=("_weekly", "_overpass"))
    corr = merged["pm25_weekly"].corr(merged["pm25_overpass"])
    print(f"wrote {out} ({len(overpass):,} station-weeks)")
    print(f"weekly vs overpass label correlation: r={corr:.3f} "
          f"(mean abs diff {(merged['pm25_weekly'] - merged['pm25_overpass']).abs().mean():.2f} ug/m3)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
