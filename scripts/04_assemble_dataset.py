#!/usr/bin/env python
"""Join labels + embeddings + station metadata -> model_table.parquet."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from thesis.config import load_config
from thesis.dataset.assemble import assemble


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--product", choices=["l2a", "l1c"], default=None)
    parser.add_argument("--labels", choices=["weekly", "overpass", "scene", "scenehour"],
                        default="weekly")
    args = parser.parse_args()

    cfg = load_config()
    product = args.product or cfg.patches["product"]
    if args.labels == "scenehour":
        # label = mean PM2.5 within +/-1h of the exact overpass timestamp
        manifest = pd.read_parquet(cfg.path("manifest"))
        scenes = manifest[(manifest["product"] == product)
                          & (manifest.get("mode", "median") == "single")
                          & (manifest["status"] == "ok")
                          & (manifest["scene_date"] != "")]
        hour_labels = pd.read_parquet(cfg.path("labels_scenehour"))
        scenes = scenes.assign(scene_date=pd.to_datetime(scenes["scene_date"]),
                               week_start=pd.to_datetime(scenes["week_start"]))
        labels = scenes[["station_id", "week_start", "scene_date"]].merge(
            hour_labels[["station_id", "scene_date", "pm25"]],
            on=["station_id", "scene_date"], how="inner")
        labels = labels[["station_id", "week_start", "pm25"]]
        print(f"scenehour labels: {len(scenes)} scenes, {len(labels)} with overpass-hour value")
        emb_name = f"embeddings_{product}_single.parquet"
    elif args.labels == "scene":
        # label = PM2.5 on the exact acquisition date of the single scene
        manifest = pd.read_parquet(cfg.path("manifest"))
        scenes = manifest[(manifest["product"] == product)
                          & (manifest.get("mode", "median") == "single")
                          & (manifest["status"] == "ok")
                          & (manifest["scene_date"] != "")]
        daily = pd.read_parquet(cfg.path("labels_daily"))
        daily["date"] = pd.to_datetime(daily["date"])
        scenes = scenes.assign(date=pd.to_datetime(scenes["scene_date"]))
        labels = scenes.merge(daily, on=["station_id", "date"], how="inner")
        labels = labels.assign(week_start=pd.to_datetime(labels["week_start"]))
        labels = labels[["station_id", "week_start", "pm25"]]
        print(f"scene labels: {len(scenes)} scenes, {len(labels)} with same-day EPA value")
        emb_name = f"embeddings_{product}_single.parquet"
    else:
        labels = pd.read_parquet(cfg.path(f"labels_{args.labels}"))
        emb_name = f"embeddings_{product}.parquet"
    embeddings = pd.read_parquet(cfg.path("embeddings").with_name(emb_name))
    stations = pd.read_parquet(cfg.path("stations"))

    table, report = assemble(labels, embeddings, stations)
    out = cfg.path("model_table").with_name(f"model_table_{product}_{args.labels}.parquet")
    table.to_parquet(out, index=False)
    out.with_suffix(".report.json").write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
