#!/usr/bin/env python
"""Scene-quality filter for the every-pass dataset.

Rules (agreed 2026-08-20):
  R1  cloud-vs-smoke: drop valid_fraction < 0.5 AND pm25 <= 20 (murk with a low
      label is cloud; murk with a high label is smoke and is kept)
  R2  single-hour labels: drop n_hours == 1
  R3  tile-edge artifacts: drop patches with > 25% all-zero pixels
  R4  low floor: drop pm25 < 2.5 (near instrument noise floor)

Writes data/interim/scene_keep.parquet (station_id, key, keep, reason).
Assembly inner-joins keep==True.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from tqdm import tqdm

from thesis.config import load_config


def main() -> int:
    cfg = load_config()
    lab = pd.read_parquet(cfg.path("labels_scenehour"))
    lab["key"] = lab["scene_date"].astype(str).str[:10]

    man = pd.read_parquet(cfg.path("manifest"))
    man = man[(man["mode"] == "scene") & (man["status"] == "ok")]
    man = man.rename(columns={"week_start": "key"})[["station_id", "key", "valid_fraction"]]

    root = cfg.path("patches_dir") / cfg.patches["product"] / "scenes"
    files = [(d.name, f.stem, f) for d in root.iterdir() if d.is_dir() for f in d.glob("*.npy")]
    df = pd.DataFrame(files, columns=["station_id", "key", "path"])
    df = df.merge(lab[["station_id", "key", "pm25", "n_hours"]], on=["station_id", "key"], how="inner")
    df = df.merge(man, on=["station_id", "key"], how="left")

    zero_frac = np.empty(len(df))
    for i, p in enumerate(tqdm(df["path"], desc="pixel scan")):
        a = np.load(p)
        zero_frac[i] = (a == 0).all(axis=-1).mean()
    df["zero_frac"] = zero_frac

    reason = np.full(len(df), "", dtype=object)
    r1 = (df["valid_fraction"].fillna(1.0) < 0.5) & (df["pm25"] <= 20)
    r2 = df["n_hours"] == 1
    r3 = df["zero_frac"] > 0.25
    r4 = df["pm25"] < 2.5
    reason[r4.to_numpy()] = "pm_below_2.5"
    reason[r3.to_numpy()] = "tile_edge_zeros"
    reason[r2.to_numpy()] = "single_hour_label"
    reason[r1.to_numpy()] = "cloud_low_label"
    df["reason"] = reason
    df["keep"] = reason == ""

    out = cfg.path("labels_scenehour").with_name("scene_keep.parquet")
    df[["station_id", "key", "keep", "reason", "valid_fraction", "zero_frac"]].to_parquet(out, index=False)
    print(f"total scenes: {len(df)}")
    print(df["reason"].replace("", "KEEP").value_counts().to_string())
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
