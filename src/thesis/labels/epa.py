"""Labels from EPA AQS pre-generated daily files (parameter 88101, PM2.5 FRM/FEM).

Source: https://aqs.epa.gov/aqsweb/airdata/daily_88101_{year}.zip
These are validated, reference-grade daily summaries — no raw QC needed here.
Remaining logic: filter to region/duration/completeness, dedupe monitors,
aggregate daily -> ISO-weekly means.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

AIRDATA_URL = "https://aqs.epa.gov/aqsweb/airdata/daily_88101_{year}.zip"

USECOLS = [
    "State Code", "County Code", "Site Num", "POC", "Latitude", "Longitude",
    "Sample Duration", "Date Local", "Event Type", "Observation Count",
    "Observation Percent", "Arithmetic Mean", "Local Site Name",
]


def download_year(year: int, dest_dir: Path) -> Path:
    dest = dest_dir / f"daily_88101_{year}.zip"
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return dest
    url = AIRDATA_URL.format(year=year)
    resp = requests.get(url, timeout=300)
    resp.raise_for_status()
    tmp = dest.with_suffix(".tmp")
    tmp.write_bytes(resp.content)
    tmp.rename(dest)
    return dest


def read_year(zip_path: Path, state_code: str) -> pd.DataFrame:
    """Parse one annual zip to a daily table for one state.

    Keeps 24-hour sample durations only (drops the redundant '1 HOUR' rows),
    keeps wildfire-influenced rows (Event Type 'Included'), and dedupes
    site-date duplicates (multiple POCs / pollutant standards) by lowest POC.
    """
    with zipfile.ZipFile(zip_path) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as f:
            df = pd.read_csv(
                io.TextIOWrapper(f, encoding="utf-8"),
                usecols=USECOLS,
                dtype={"State Code": str, "County Code": str, "Site Num": str},
                low_memory=False,
            )
    df = df[df["State Code"] == state_code]
    # Keep 24-hour daily values: "24 HOUR" (FRM samplers) and "24-HR BLK AVG"
    # (daily averages of continuous FEM monitors). "1 HOUR" rows duplicate the
    # latter. Event Type is blank for most rows; drop only explicit "Excluded".
    df = df[df["Sample Duration"].str.contains("24", na=False)]
    df = df[df["Event Type"].ne("Excluded")]

    df["station_id"] = (
        df["State Code"].str.zfill(2)
        + "-" + df["County Code"].str.zfill(3)
        + "-" + df["Site Num"].str.zfill(4)
    )
    df["date"] = pd.to_datetime(df["Date Local"])
    df = df.sort_values("POC").drop_duplicates(subset=["station_id", "date"], keep="first")
    # FEM monitors report small negatives on very clean days (instrument noise
    # around zero); clip to 0 so downstream log transforms are safe
    df["Arithmetic Mean"] = df["Arithmetic Mean"].clip(lower=0.0)
    return df.rename(columns={
        "Latitude": "lat", "Longitude": "lon",
        "Arithmetic Mean": "pm25", "Observation Percent": "obs_percent",
        "Local Site Name": "site_name",
    })[["station_id", "date", "pm25", "obs_percent", "lat", "lon", "site_name"]]


def build_stations(daily: pd.DataFrame) -> pd.DataFrame:
    """One row per station with representative coordinates and day counts."""
    return (
        daily.groupby("station_id")
        .agg(lat=("lat", "median"), lon=("lon", "median"),
             site_name=("site_name", "first"), n_days=("pm25", "size"))
        .reset_index()
    )


def aggregate_weekly(
    daily: pd.DataFrame,
    min_observation_percent: float = 75.0,
    min_days_per_week: int = 5,
) -> pd.DataFrame:
    """Daily -> ISO-week means. Week key is the Monday date of the ISO week."""
    d = daily[daily["obs_percent"] >= min_observation_percent].copy()
    d["week_start"] = (d["date"] - pd.to_timedelta(d["date"].dt.dayofweek, unit="D")).dt.normalize()
    wk = (
        d.groupby(["station_id", "week_start"])
        .agg(pm25=("pm25", "mean"), n_days=("pm25", "size"))
        .reset_index()
    )
    return wk[wk["n_days"] >= min_days_per_week].reset_index(drop=True)


def filter_station_coverage(weekly: pd.DataFrame, years: list[int], min_coverage: float) -> pd.DataFrame:
    """Keep stations covering >= min_coverage of all ISO weeks in the listed years.

    Years may be non-contiguous (e.g. [2020, 2023, 2024]); only weeks belonging
    to listed years count toward the denominator.
    """
    total_weeks = sum(
        len(pd.date_range(f"{y}-01-01", f"{y}-12-31", freq="W-MON")) for y in years
    )
    counts = weekly.groupby("station_id")["week_start"].nunique()
    keep = counts[counts >= min_coverage * total_weeks].index
    return weekly[weekly["station_id"].isin(keep)].reset_index(drop=True)


def build_overpass_labels(daily: pd.DataFrame, pass_dates: pd.DataFrame) -> pd.DataFrame:
    """Label each station-week by PM2.5 on the actual Sentinel-2 overpass day(s).

    daily:      station_id, date, pm25 (local calendar dates)
    pass_dates: station_id, date (local dates with an S2 acquisition)

    The weekly-mean label mixes days the satellite never saw; here the label is
    the mean over only the overpass days that have an EPA measurement.
    """
    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    p = pass_dates.drop_duplicates(["station_id", "date"]).copy()
    p["date"] = pd.to_datetime(p["date"]).dt.normalize()

    m = p.merge(d, on=["station_id", "date"], how="inner")
    m["week_start"] = (m["date"] - pd.to_timedelta(m["date"].dt.dayofweek, unit="D"))
    out = (
        m.groupby(["station_id", "week_start"])
        .agg(pm25=("pm25", "mean"), n_days=("date", "nunique"))
        .reset_index()
    )
    return out
