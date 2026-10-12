#!/usr/bin/env python
"""Re-score the final ensemble on a label-blind test set (log §3ae).

Two of the four quality rules look at the label (cloud-vs-smoke keeps murky scenes only when
the label is high; the low-label rule drops scenes below 2.5 ug/m3). At an unmonitored location
neither can be applied. This script runs the five ensemble members on the test-station scenes
that those two rules removed, adds them to the saved predictions, and reports the metrics on
the label-blind set (kept + cloud_low_label + pm_below_2.5; the tile-edge and single-hour rules
do not use the label value).
"""
import importlib.util, sys, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("ft", ROOT / "scripts/07_finetune.py"); ft = importlib.util.module_from_spec(spec); spec.loader.exec_module(ft)
from thesis.config import load_config
cfg = load_config(); dev = "mps" if torch.backends.mps.is_available() else "cpu"
ALL = list(ft.ALL_CTX_COLS); NOLL = [c for c in ALL if c not in ("lat", "lon")]; GEO = ["lat", "lon", "doy_sin", "doy_cos"]
MEMBERS = {"A": [("fused_film", ALL, True, None), ("fused_film_nolatlon", NOLL, True, None), ("fused_film_geotime", GEO, True, None),
                 ("film_full_seed2", ALL, False, None), ("film_reftemporal_s0", ALL, False, "temporal")],
           "B": [("reval_film_full", ALL, True, None), ("reval_film_nolatlon", NOLL, True, None), ("reval_film_geotime", GEO, True, None),
                 ("film_full_seed2_B", ALL, False, None), ("film_reftemporal_B", ALL, False, "temporal")]}
PREDS = {"A": ["tta_film_s0", "tta_nolatlon_s0", "tta_geotime_s0", "film_full_seed2", "film_reftemporal_s0"],
         "B": ["tta_film_s1", "tta_nolatlon_s1", "tta_geotime_s1", "film_full_seed2_B", "film_reftemporal_B"]}
def r2(y, p): return 1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()
def auc(y, p):
    o = np.argsort(-p); yy = y[o]; tp = np.cumsum(yy); fp = np.cumsum(~yy); return np.trapezoid(np.r_[0, tp / yy.sum()], np.r_[0, fp / (~yy).sum()])
def metrics(d):
    y, p = d.y_true.values, d.y_pred.values; t, q = y > 35, p > 35
    return dict(n=len(d), r2=round(r2(y, p), 3), mae=round(np.abs(p - y).mean(), 2), auc=round(auc(t, p), 3),
                precision=round((t & q).sum() / max(q.sum(), 1), 2), recall=round((t & q).sum() / max(t.sum(), 1), 2), false_alarms=int((~t & q).sum()), exceed=int(t.sum()))

table = pd.read_parquet(cfg.path("model_table").with_name("model_table_l1c_scenehour_allscenes.parquet"))[["station_id", "week_start", "pm25", "region"]]
ctx = pd.read_parquet(cfg.path("labels_scenehour").with_name("context_features.parquet")); ctx["week_start"] = pd.to_datetime(ctx["key"])
table["week_start"] = pd.to_datetime(table["week_start"]); table = table.merge(ctx.drop(columns=["key"]), on=["station_id", "week_start"], how="left").dropna(subset=ALL).reset_index(drop=True)
keep = pd.read_parquet(cfg.path("labels_scenehour").with_name("scene_keep.parquet")); keep["week_start"] = pd.to_datetime(keep["key"])
lab = pd.read_parquet(cfg.path("labels_scenehour")); lab["week_start"] = pd.to_datetime(lab["scene_date"])
rej = keep[keep.reason.isin(["cloud_low_label", "pm_below_2.5"])].merge(lab[["station_id", "week_start", "pm25"]], on=["station_id", "week_start"])
rej = rej.merge(ctx.drop(columns=["key"]), on=["station_id", "week_start"], how="left").dropna(subset=ALL)
patch_root = cfg.path("patches_dir") / "l1c" / "scenes"; out = {}
for nm, seed in [("A", 0), ("B", 1)]:
    tr_idx, te_idx = next(ft.iter_splits(table, "holdout", seed)); test = set(table.iloc[te_idx].station_id); train = table.iloc[tr_idx]
    saved = ft.__dict__  # noqa
    extra = rej[rej.station_id.isin(test)].reset_index(drop=True); extra["key"] = extra.week_start.dt.strftime("%Y-%m-%d")
    print(f"split {nm}: {len(test)} test stations, {len(extra)} rejected scenes to score ({extra.reason.value_counts().to_dict()})", flush=True)
    preds = []
    for run, cols, tta, ref in MEMBERS[nm]:
        ft.CTX_COLS[:] = cols
        cm, cs = train[cols].mean().to_numpy("float32"), train[cols].std().to_numpy("float32") + 1e-6
        rows = extra.rename(columns={}).copy(); rows["week_start"] = rows["key"]   # dataset expects the date string
        ds = ft.PatchDataset(rows, patch_root, cfg.embeddings["rgb_gain"], False, ctx_stats=(cm, cs), ref_mode=ref)
        model = ft.make_fused(6 if ref else 3, len(cols), "film").to(dev)
        ck = torch.load(ROOT / f"data/runs/{run}/model_holdout_s{seed}.pt", map_location=dev)
        model.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
        p = ft.predict(model, DataLoader(ds, batch_size=64, num_workers=0), dev, tta=tta); preds.append(p)
        print(f"   {run}: scored {len(p)} scenes (tta={tta})", flush=True)
    extra["y_pred"] = np.mean(preds, axis=0); extra["y_true"] = extra.pm25
    kept = None
    for i, run in enumerate(PREDS[nm]):
        d = pd.read_parquet(ROOT / f"data/runs/{run}/preds_holdout_f0.parquet"); d["week_start"] = pd.to_datetime(d.week_start)
        kept = d[["station_id", "week_start", "y_true"]].copy() if kept is None else kept
        kept[f"m{i}"] = kept.set_index(["station_id", "week_start"]).index.map(d.set_index(["station_id", "week_start"]).y_pred)
    kept = kept.dropna(); kept["y_pred"] = kept[[f"m{i}" for i in range(5)]].mean(axis=1)
    blind = pd.concat([kept[["station_id", "week_start", "y_true", "y_pred"]].assign(reason=""), extra[["station_id", "week_start", "y_true", "y_pred", "reason"]]])
    extra.to_parquet(ROOT / f"data/runs/labelblind_extra_{nm}.parquet", index=False)
    out[nm] = {"kept (thesis)": metrics(kept), "label-blind (kept + cloud_low_label + pm_below_2.5)": metrics(blind),
               "kept + cloud_low_label only": metrics(blind[blind.reason != "pm_below_2.5"]), "rejected scenes alone": metrics(extra)}
    for k, v in out[nm].items(): print(f"   {k}: {v}", flush=True)
json.dump(out, open(ROOT / "data/runs/labelblind_summary.json", "w"), indent=1)
