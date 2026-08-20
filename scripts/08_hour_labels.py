#!/usr/bin/env python
"""Timestamp-level label sync: PM2.5 in a +/-1h window around the exact S2 overpass.

1. Download EPA hourly files (hourly_88101_{year}.zip) and parse CA FEM monitors.
2. Query exact S2 acquisition timestamps per station (kept to the minute,
   converted to local time) -> s2_pass_times.parquet.
3. Label = mean hourly PM2.5 within +/-1 hour of overpass -> labels_scenehour.parquet.
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
from thesis.labels import epa

def query_pass_times(cfg, stations: pd.DataFrame) -> pd.DataFrame:
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
        t = (
            pd.to_datetime(pd.Series(ts, dtype="int64"), unit="ms", utc=True)
            .dt.tz_localize(None)  # keep UTC, naive
            .drop_duplicates()
        )
        return pd.DataFrame({"station_id": row.station_id, "ts": t})

    frames = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(one, row) for row in stations.itertuples()]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="pass times"):
            frames.append(fut.result())
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--window-hours", type=int, default=1)
    args = parser.parse_args()

    cfg = load_config()

    times_cache = cfg.path("s2_pass_times")
    stations = pd.read_parquet(cfg.path("stations"))
    if times_cache.exists() and not args.force:
        pass_times = pd.read_parquet(times_cache)
        if "ts_local" in pass_times.columns:  # pre-UTC cache: rebuild
            pass_times = pass_times.iloc[0:0]
            pass_times = pd.DataFrame(columns=["station_id", "ts"])
        missing = stations[~stations["station_id"].isin(pass_times["station_id"])]
        if len(missing):
            init_ee(cfg.ee_project)
            print(f"querying pass times for {len(missing)} new stations")
            pass_times = pd.concat([pass_times, query_pass_times(cfg, missing)],
                                   ignore_index=True)
            pass_times.to_parquet(times_cache, index=False)
    else:
        init_ee(cfg.ee_project)
        pass_times = query_pass_times(cfg, stations)
        pass_times.to_parquet(times_cache, index=False)
    print(f"pass times: {len(pass_times):,} acquisitions "
          f"(hour-of-day range {pass_times['ts'].dt.hour.min()}-"
          f"{pass_times['ts'].dt.hour.max()} UTC)")

    hourly_parts = []
    for year in cfg.years:
        zp = epa.download_hourly_year(year, cfg.path("raw_epa"))
        for reg in cfg.regions:
            part = epa.read_hourly_year(zp, reg["epa_state_code"])
            print(f"{year} {reg['name']}: {len(part):,} station-hours, "
                  f"{part['station_id'].nunique()} stations")
            hourly_parts.append(part)
    hourly = pd.concat(hourly_parts, ignore_index=True)

    labels = epa.build_overpass_hour_labels(hourly, pass_times, args.window_hours)
    out = cfg.path("labels_scenehour")
    labels.to_parquet(out, index=False)
    print(f"wrote {out} ({len(labels):,} station-scenes, "
          f"{labels['station_id'].nunique()} stations)")

    daily = pd.read_parquet(cfg.path("labels_daily"))
    daily["date"] = pd.to_datetime(daily["date"])
    m = labels.merge(daily, left_on=["station_id", "scene_date"],
                     right_on=["station_id", "date"], suffixes=("_hour", "_daily"))
    print(f"overpass-hour vs same-day daily-mean correlation: "
          f"r={m['pm25_hour'].corr(m['pm25_daily']):.3f} "
          f"(mean abs diff {(m['pm25_hour'] - m['pm25_daily']).abs().mean():.2f} ug/m3)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
