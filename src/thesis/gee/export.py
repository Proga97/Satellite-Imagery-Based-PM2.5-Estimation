"""Threaded computePixels download with retry/backoff, manifest, and resume."""
from __future__ import annotations

import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path

import ee
import numpy as np
import pandas as pd
from tqdm import tqdm

from .patches import build_composite, parse_pixels

RETRYABLE = ("429", "quota", "rate", "unavailable", "deadline", "timeout", "internal", "500", "503")


@dataclass
class PatchJob:
    station_id: str
    week_start: str  # ISO date (Monday)
    lat: float
    lon: float
    epsg: int


@dataclass
class PatchResult:
    station_id: str
    week_start: str
    product: str
    status: str  # ok | too_cloudy | no_images | error
    valid_fraction: float = float("nan")
    n_images: int = 0
    error: str = ""
    mode: str = "median"
    scene_date: str = ""
    bands: str = "rgb"


def patch_path(patches_dir: Path, product: str, station_id: str, week_start: str,
               mode: str = "median", band_set: str = "rgb") -> Path:
    subdir = {"median": "weekly", "single": "single", "scene": "scenes"}[mode]
    if band_set != "rgb":
        subdir = f"{subdir}_{band_set}"
    return patches_dir / product / subdir / station_id / f"{week_start}.npy"


def _download_one(job: PatchJob, cfg_patches: dict, patches_dir: Path) -> PatchResult:
    product = cfg_patches["product"]
    mode = cfg_patches.get("mode", "median")
    band_set = cfg_patches.get("band_set", "rgb")
    bands = list(cfg_patches["bands"])
    start = job.week_start
    window_days = 1 if mode == "scene" else 7
    end = (pd.Timestamp(job.week_start) + pd.Timedelta(days=window_days)).strftime("%Y-%m-%d")

    comp, grid = build_composite(job.lat, job.lon, start, end, cfg_patches, job.epsg)
    request = {"expression": comp, "fileFormat": "NUMPY_NDARRAY", "grid": grid.to_request_grid()}

    last_err = ""
    for attempt in range(cfg_patches["max_retries"]):
        try:
            arr = ee.data.computePixels(request)
            rgb, valid_fraction, n_images, scene_date = parse_pixels(arr, bands)
            if n_images == 0:
                return PatchResult(job.station_id, job.week_start, product, "no_images", mode=mode, bands=band_set)
            if valid_fraction < cfg_patches["min_valid_fraction"]:
                return PatchResult(job.station_id, job.week_start, product, "too_cloudy",
                                   valid_fraction, n_images, mode=mode, scene_date=scene_date,
                                   bands=band_set)
            out = patch_path(patches_dir, product, job.station_id, job.week_start, mode, band_set)
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(".tmp.npy")
            np.save(tmp, rgb)
            tmp.rename(out)
            return PatchResult(job.station_id, job.week_start, product, "ok",
                               valid_fraction, n_images, mode=mode, scene_date=scene_date,
                               bands=band_set)
        except Exception as exc:
            last_err = str(exc)
            # empty collection: median() has no bands / first() is null ->
            # a week with zero images, not a real error
            if "has no bands" in last_err or "Parameter 'image' is required" in last_err:
                return PatchResult(job.station_id, job.week_start, product, "no_images", mode=mode, bands=band_set)
            if any(k in last_err.lower() for k in RETRYABLE):
                time.sleep(min(60.0, (2 ** attempt) + random.random()))
                continue
            break
    return PatchResult(job.station_id, job.week_start, product, "error",
                       error=last_err[:300], mode=mode, bands=band_set)


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    if manifest_path.exists():
        return pd.read_parquet(manifest_path)
    return pd.DataFrame(columns=[f.name for f in PatchResult.__dataclass_fields__.values()])


def run_downloads(jobs: list[PatchJob], cfg_patches: dict, patches_dir: Path,
                  manifest_path: Path) -> pd.DataFrame:
    """Download all jobs not already resolved; merge results into the manifest."""
    product = cfg_patches["product"]
    mode = cfg_patches.get("mode", "median")
    band_set = cfg_patches.get("band_set", "rgb")
    manifest = load_manifest(manifest_path)
    if len(manifest) and "mode" not in manifest.columns:
        manifest["mode"] = "median"
        manifest["scene_date"] = ""
    if len(manifest) and "bands" not in manifest.columns:
        manifest["bands"] = "rgb"
    done_keys = set()
    if len(manifest):
        prior = manifest[(manifest["product"] == product)
                         & (manifest["mode"].fillna("median") == mode)
                         & (manifest["bands"].fillna("rgb") == band_set)
                         & (manifest["status"].isin(["ok", "too_cloudy", "no_images"]))]
        done_keys = set(zip(prior["station_id"], prior["week_start"]))

    todo = []
    for j in jobs:
        if (j.station_id, j.week_start) in done_keys:
            continue
        if patch_path(patches_dir, product, j.station_id, j.week_start, mode, band_set).exists():
            continue
        todo.append(j)

    results: list[PatchResult] = []
    if todo:
        with ThreadPoolExecutor(max_workers=cfg_patches["workers"]) as pool:
            futures = {pool.submit(_download_one, j, cfg_patches, patches_dir): j for j in todo}
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"patches[{product}]"):
                results.append(fut.result())

    new = pd.DataFrame([asdict(r) for r in results])
    merged = pd.concat([manifest, new], ignore_index=True)
    merged = merged.drop_duplicates(subset=["station_id", "week_start", "product", "mode", "bands"], keep="last")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(manifest_path, index=False)
    return merged
