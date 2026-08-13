#!/usr/bin/env python
"""Train + evaluate. Phase 1: image-only, random + spatial splits."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from thesis.config import load_config
from thesis.dataset.assemble import feature_columns
from thesis.models.metrics import compute_metrics
from thesis.models.splits import SPLITS
from thesis.models.train import fit_predict


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="phase1_image_only")
    parser.add_argument("--feature-sets", nargs="+", default=["image"])
    parser.add_argument("--models", nargs="+", default=["lightgbm", "rf"])
    parser.add_argument("--splits", nargs="+", default=["random", "spatial"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--test-year", type=int, default=2024)
    args = parser.parse_args()

    cfg = load_config()
    table = pd.read_parquet(cfg.path("model_table"))
    print(f"model table: {len(table)} rows, {table['station_id'].nunique()} stations")

    run_dir = cfg.path("runs_dir") / args.experiment
    run_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for feature_set in args.feature_sets:
        feat_cols = feature_columns(table, feature_set)
        for model_name in args.models:
            for split_name in args.splits:
                split_fn = SPLITS[split_name]
                split_kwargs = {"test_year": args.test_year} if split_name == "temporal" else {}
                for seed in args.seeds:
                    fold_metrics = []
                    preds_parts = []
                    for fold, (tr, te) in enumerate(split_fn(table, **split_kwargs) if split_kwargs
                                                    else split_fn(table, seed=seed) if split_name == "random"
                                                    else split_fn(table)):
                        y_pred = fit_predict(model_name, seed, table, feat_cols, tr, te)
                        y_true = table["pm25"].to_numpy()[te]
                        sids = table["station_id"].to_numpy()[te]
                        m = compute_metrics(y_true, y_pred, sids)
                        m["fold"] = fold
                        fold_metrics.append(m)
                        preds_parts.append(pd.DataFrame({
                            "station_id": sids, "week_start": table["week_start"].to_numpy()[te],
                            "y_true": y_true, "y_pred": y_pred, "fold": fold,
                        }))
                    agg = {k: float(np.mean([fm[k] for fm in fold_metrics if k in fm]))
                           for k in fold_metrics[0] if k != "fold"}
                    row = {"feature_set": feature_set, "model": model_name,
                           "split": split_name, "seed": seed, "n_folds": len(fold_metrics), **agg}
                    all_rows.append(row)
                    tag = f"{feature_set}_{model_name}_{split_name}_s{seed}"
                    pd.concat(preds_parts).to_parquet(run_dir / f"preds_{tag}.parquet", index=False)
                    print(f"{tag}: r2={agg.get('r2', float('nan')):.3f} "
                          f"rmse={agg.get('rmse', float('nan')):.2f} "
                          f"between={agg.get('between_station_r2', float('nan')):.3f} "
                          f"within={agg.get('within_station_r2', float('nan')):.3f}")

    results = pd.DataFrame(all_rows)
    results.to_parquet(run_dir / "results.parquet", index=False)
    (run_dir / "results.json").write_text(json.dumps(all_rows, indent=2))

    print("\n== summary (mean over seeds) ==")
    summary = results.groupby(["feature_set", "model", "split"])[
        [c for c in ["r2", "rmse", "mae", "between_station_r2", "within_station_r2"] if c in results]
    ].mean().round(3)
    print(summary.to_string())
    print(f"\nwrote {run_dir}/results.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
