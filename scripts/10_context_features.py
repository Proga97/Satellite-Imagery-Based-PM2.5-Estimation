#!/usr/bin/env python
"""Context features for every labeled scene: ERA5 weather, elevation, sun angle, season.

All inputs are available for ANY point on Earth (no ground infrastructure), so the
"works where there is no monitor" story survives:
  - ERA5-Land daily aggregates (GEE): temp, dewpoint (-> RH), wind u/v (-> speed),
    precipitation, surface pressure -- one getRegion per station over the full period.
  - SRTM elevation at the station point.
  - Solar elevation at the exact overpass timestamp (computed astronomically, no data).
  - Day-of-year sin/cos.

Output: data/interim/context_features.parquet keyed (station_id, key=scene date ISO).
"""
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from tqdm import tqdm

from thesis.config import load_config
from thesis.gee.auth import init_ee

ERA5 = "ECMWF/ERA5_LAND/DAILY_AGGR"
BANDS = ["temperature_2m", "dewpoint_temperature_2m", "u_component_of_wind_10m",
         "v_component_of_wind_10m", "total_precipitation_sum", "surface_pressure"]


def solar_elevation_deg(lat, lon, ts_utc):
    """Approximate solar elevation (degrees) at UTC timestamp; NOAA-style formula."""
    ts = pd.to_datetime(ts_utc)
    doy = ts.dayofyear
    frac_hour = ts.hour + ts.minute / 60.0
    gamma = 2 * np.pi / 365 * (doy - 1 + (frac_hour - 12) / 24)
    decl = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
            - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
            - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma))
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
                       - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma))
    tst = frac_hour * 60 + eqtime + 4 * lon
    ha = np.deg2rad(tst / 4 - 180)
    lat_r = np.deg2rad(lat)
    cos_zen = (np.sin(lat_r) * np.sin(decl) + np.cos(lat_r) * np.cos(decl) * np.cos(ha))
    return float(np.rad2deg(np.arcsin(np.clip(cos_zen, -1, 1))))


def main() -> int:
    cfg = load_config()
    lab = pd.read_parquet(cfg.path("labels_scenehour"))
    lab["key"] = lab["scene_date"].astype(str).str[:10]
    st = pd.read_parquet(cfg.path("stations")).set_index("station_id")
    pt = pd.read_parquet(cfg.path("s2_pass_times"))
    pt["key"] = pt["ts"].dt.strftime("%Y-%m-%d")
    pass_ts = pt.drop_duplicates(["station_id", "key"]).set_index(["station_id", "key"])["ts"]

    init_ee(cfg.ee_project)
    import ee

    start, end = f"{min(cfg.years)}-01-01", f"{max(cfg.years) + 1}-01-01"
    stations = st.reset_index()

    # SRTM elevation for all stations in one query
    srtm = ee.Image("USGS/SRTMGL1_003")
    pts = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([r.lon, r.lat]), {"sid": r.station_id})
        for r in stations.itertuples()])
    elev_info = srtm.reduceRegions(pts, ee.Reducer.first(), 90).getInfo()
    elev = {f["properties"]["sid"]: f["properties"].get("first", 0.0)
            for f in elev_info["features"]}
    print(f"elevation fetched for {len(elev)} stations")

    def one(row):
        # per-year chunks: a 6-year getRegion exceeds GEE's per-request memory cap
        point = ee.Geometry.Point([row.lon, row.lat])
        parts = []
        for y in cfg.years:
            col = (ee.ImageCollection(ERA5)
                   .filterDate(f"{y}-01-01", f"{y + 1}-01-01").select(BANDS))
            arr = col.getRegion(point, 11132).getInfo()
            df = pd.DataFrame(arr[1:], columns=arr[0])
            parts.append(df)
        df = pd.concat(parts, ignore_index=True)
        df["date"] = pd.to_datetime(df["time"], unit="ms").dt.strftime("%Y-%m-%d")
        df["station_id"] = row.station_id
        return df[["station_id", "date"] + BANDS]

    frames = []
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(one, r) for r in stations.itertuples()]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="ERA5"):
            frames.append(fut.result())
    met = pd.concat(frames, ignore_index=True)
    print(f"ERA5 rows: {len(met):,}")

    # derived met variables
    t = met["temperature_2m"] - 273.15
    td = met["dewpoint_temperature_2m"] - 273.15
    met["rh"] = 100 * (np.exp(17.625 * td / (243.04 + td)) / np.exp(17.625 * t / (243.04 + t)))
    met["wind_speed"] = np.hypot(met["u_component_of_wind_10m"], met["v_component_of_wind_10m"])
    met["temp_c"] = t
    met["precip_mm"] = met["total_precipitation_sum"] * 1000
    met["pressure_hpa"] = met["surface_pressure"] / 100

    scenes = lab[["station_id", "key"]].drop_duplicates()
    out = scenes.merge(met[["station_id", "date", "temp_c", "rh", "wind_speed",
                            "precip_mm", "pressure_hpa"]],
                       left_on=["station_id", "key"], right_on=["station_id", "date"],
                       how="left").drop(columns=["date"])
    out["elevation_m"] = out["station_id"].map(elev)
    lat = out["station_id"].map(st["lat"]); lon = out["station_id"].map(st["lon"])
    ts = [pass_ts.get((s, k)) for s, k in zip(out["station_id"], out["key"])]
    out["sun_elev_deg"] = [solar_elevation_deg(la, lo, t) if t is not None else np.nan
                           for la, lo, t in zip(lat, lon, ts)]
    doy = pd.to_datetime(out["key"]).dt.dayofyear
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    out["lat"] = lat.values
    out["lon"] = lon.values

    path = cfg.path("labels_scenehour").with_name("context_features.parquet")
    out.to_parquet(path, index=False)
    print(f"wrote {path} ({len(out):,} scene-contexts, "
          f"met missing: {out['temp_c'].isna().sum()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
