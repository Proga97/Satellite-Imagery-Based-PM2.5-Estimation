#!/usr/bin/env python
"""Final step of the data pipeline: one row per usable scene.

Joins the downloaded scenes (02), the overpass-hour labels (08), the quality decisions (09)
and the context features (10) into a single table that needs nothing else to be used:

    station_id, scene_date, pass_time_utc, pm25, n_hours, lat, lon, region, patch_path,
    valid_fraction, temp_c, rh, wind_speed, precip_mm, pressure_hpa, elevation_m,
    sun_elev_deg, doy_sin, doy_cos

Writes data/processed/dataset.parquet and dataset.csv (same rows), plus the table that
07_finetune.py reads (model_table_<product>_scenehour_allscenes.parquet, same rows with
the column names the training script expects). Scenes that failed a quality rule are
excluded; context columns are NaN for the few coastal stations outside ERA5-Land.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd  # noqa: E402

from thesis.config import load_config  # noqa: E402

CTX = ["temp_c", "rh", "wind_speed", "precip_mm", "pressure_hpa", "elevation_m",
       "sun_elev_deg", "doy_sin", "doy_cos"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--product", choices=["l2a", "l1c"], default=None,
                        help="override configs/pipeline.yaml patches.product")
    parser.add_argument("--keep-rejected", action="store_true",
                        help="keep scenes that failed a quality rule (column 'reason' says why)")
    args = parser.parse_args()
    cfg = load_config()
    product = args.product or cfg.patches["product"]

    manifest = pd.read_parquet(cfg.path("manifest"))
    scenes = manifest[(manifest["product"] == product) & (manifest["mode"] == "scene")
                      & (manifest["bands"].fillna("rgb") == "rgb") & (manifest["status"] == "ok")
                      & (manifest["scene_date"] != "")]
    scenes = scenes[["station_id", "scene_date", "valid_fraction"]].copy()
    scenes["scene_date"] = pd.to_datetime(scenes["scene_date"])
    print(f"{len(scenes)} downloaded {product} scenes")

    labels = pd.read_parquet(cfg.path("labels_scenehour"))
    labels["scene_date"] = pd.to_datetime(labels["scene_date"])
    rows = scenes.merge(labels[["station_id", "scene_date", "pm25", "n_hours"]],
                        on=["station_id", "scene_date"], how="inner")
    print(f"{len(rows)} with a PM2.5 value within one hour of the pass")

    keep_path = cfg.path("labels_scenehour").with_name("scene_keep.parquet")
    if keep_path.exists():
        keep = pd.read_parquet(keep_path)
        keep["scene_date"] = pd.to_datetime(keep["key"])
        rows = rows.merge(keep[["station_id", "scene_date", "keep", "reason"]],
                          on=["station_id", "scene_date"], how="left")
        rows["keep"] = rows["keep"].fillna(True)
        if not args.keep_rejected:
            n0 = len(rows); rows = rows[rows["keep"]].drop(columns=["keep", "reason"])
            print(f"{n0 - len(rows)} removed by the quality rules (09_clean_scenes.py)")
    else:
        print("no scene_keep.parquet: run 09_clean_scenes.py to apply the quality rules")

    pt = pd.read_parquet(cfg.path("s2_pass_times"))
    pt["scene_date"] = pd.to_datetime(pt["ts"].dt.strftime("%Y-%m-%d"))
    pt = pt.drop_duplicates(["station_id", "scene_date"]).rename(columns={"ts": "pass_time_utc"})
    rows = rows.merge(pt[["station_id", "scene_date", "pass_time_utc"]],
                      on=["station_id", "scene_date"], how="left")

    stations = pd.read_parquet(cfg.path("stations"))[["station_id", "lat", "lon", "region"]]
    rows = rows.merge(stations, on="station_id", how="left")

    ctx_path = cfg.path("labels_scenehour").with_name("context_features.parquet")
    if ctx_path.exists():
        ctx = pd.read_parquet(ctx_path)
        ctx["scene_date"] = pd.to_datetime(ctx["key"])
        rows = rows.merge(ctx[["station_id", "scene_date"] + CTX], on=["station_id", "scene_date"], how="left")
        print(f"{rows[CTX].isna().any(axis=1).sum()} scenes without context (outside ERA5-Land)")
    else:
        print("no context_features.parquet: run 10_context_features.py to add weather and elevation")

    patch_dir = cfg.path("patches_dir") / product / "scenes"
    rows["patch_path"] = [str((patch_dir / s / f"{d:%Y-%m-%d}.npy").relative_to(cfg.root))
                          for s, d in zip(rows["station_id"], rows["scene_date"])]
    rows = rows.sort_values(["station_id", "scene_date"]).reset_index(drop=True)

    out = cfg.path("model_table").parent
    cols = ["station_id", "scene_date", "pass_time_utc", "pm25", "n_hours", "lat", "lon", "region",
            "patch_path", "valid_fraction"] + [c for c in CTX if c in rows] + (["reason"] if args.keep_rejected else [])
    rows[cols].to_parquet(out / "dataset.parquet", index=False)
    rows[cols].to_csv(out / "dataset.csv", index=False)
    # the training script's table: same scenes, its column names
    mt = rows.rename(columns={"scene_date": "week_start"})[["station_id", "week_start", "pm25", "lat", "lon", "region"]]
    mt["year"] = mt["week_start"].dt.year
    mt_path = out / f"model_table_{product}_scenehour_allscenes.parquet"
    if mt_path.exists():   # keep the existing table: the row order fixes the validation carve-out of past runs
        print(f"{mt_path.name} exists; left unchanged (delete it to regenerate)")
    else:
        mt.to_parquet(mt_path, index=False)
    print(f"wrote {out / 'dataset.parquet'} ({len(rows)} scenes, {rows.station_id.nunique()} stations) and .csv; "
          f"training table model_table_{product}_scenehour_allscenes.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
