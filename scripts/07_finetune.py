#!/usr/bin/env python
"""Fine-tune ResNet-18 end-to-end on PM2.5 (the Zheng 2020 recipe, honest splits).

Frozen ImageNet features are trained to be invariant to color/contrast shifts -
exactly the visual signature of haze. Fine-tuning lets the network learn
haze-sensitive features directly from PM labels.

Augmentation: flips/rotations only. NO color jitter (it would destroy the
atmospheric signal we want the network to learn).

Evaluation mirrors 05_train_eval: random split and leave-station-out spatial
split, each trained from scratch so no leakage across protocols.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from thesis.config import load_config
from thesis.models.metrics import compute_metrics
from thesis.models.splits import random_split, region_split, spatial_folds


def iter_splits(table: pd.DataFrame, name: str, seed: int):
    if name == "final":
        # train on ALL stations (no test holdout): the deployable model.
        # test set = empty; early stopping still uses a station-aware val carve-out.
        yield np.arange(len(table)), np.array([], dtype=int)
    elif name == "holdout":
        # standard split: 80% of stations train, 20% test - STRATIFIED by region
        # so small states keep 80% training representation (an unstratified draw
        # put 35% of WA stations in test and its R2 collapsed to -2.7)
        rng = np.random.default_rng(seed)
        test_st: set = set()
        by_region = table.groupby("region")["station_id"] if "region" in table else             {"all": table["station_id"]}.items()
        for _, ids in (by_region if isinstance(by_region, list) else by_region):
            ids = ids.unique()
            rng.shuffle(ids)
            test_st.update(ids[: max(1, int(len(ids) * 0.2))])
        mask = table["station_id"].isin(test_st).to_numpy()
        yield np.flatnonzero(~mask), np.flatnonzero(mask)
    elif name == "random":
        yield from random_split(table, seed=seed)
    elif name == "spatial":
        yield from spatial_folds(table)
    elif name == "region":
        yield from region_split(table)
    else:
        raise ValueError(f"unsupported split for finetune: {name}")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


CTX_COLS = ["lat", "lon", "elevation_m", "temp_c", "rh", "wind_speed", "precip_mm",
            "pressure_hpa", "sun_elev_deg", "doy_sin", "doy_cos"]
# frozen full list: sample filtering always uses this so every fused experiment
# trains on the same 52,315 scenes / same stratified split regardless of ctx subset
ALL_CTX_COLS = list(CTX_COLS)
# optional extras selectable via --ctx-cols (scene-mean band ratios; haze scatters blue)
EXTRA_CTX_COLS = ["blue_red", "green_red"]


class PatchDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, patch_root: Path, gain: float, train: bool,
                 band_stats: tuple | None = None, classify_threshold: float | None = None,
                 ctx_stats: tuple | None = None, ref_mode: str | None = None):
        self.rows = rows.reset_index(drop=True)
        self.root = patch_root
        self.gain = gain
        self.train = train
        self.band_stats = band_stats  # (mean[C], std[C]) in reflectance units, or None for RGB path
        self.classify_threshold = classify_threshold
        self.ctx_stats = ctx_stats
        self.ref_mode = ref_mode
        self._ref_cache: dict = {}
        if ctx_stats is not None:
            cm, cs = ctx_stats
            self.ctx = ((self.rows[CTX_COLS].to_numpy(dtype=np.float32) - cm) / cs)
        else:
            self.ctx = np.zeros((len(self.rows), 0), dtype=np.float32)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows.iloc[i]
        wk = pd.Timestamp(r["week_start"]).strftime("%Y-%m-%d")
        arr = np.load(self.root / r["station_id"] / f"{wk}.npy").astype(np.float32)
        if self.band_stats is not None:
            refl = arr / 10000.0
            mean, std = self.band_stats
            img = (refl - mean) / std
        else:
            img = np.clip(arr / 10000.0 * self.gain, 0, 1)
            img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if self.ref_mode:
            sid = r["station_id"]
            ref = self._ref_cache.get(sid)
            if ref is None:
                ra = np.load(self.root.parent / "refs" / f"{sid}_{self.ref_mode}.npy")
                ref = np.clip(ra.astype(np.float32) / 10000.0 * self.gain, 0, 1)
                ref = (ref - IMAGENET_MEAN) / IMAGENET_STD
                ref = torch.from_numpy(ref.transpose(2, 0, 1))
                self._ref_cache[sid] = ref
            # channels 4-6: normalized difference today-minus-reference (surface cancels)
            img = torch.cat([img, img - ref], dim=0)
        if self.train:
            if torch.rand(1) < 0.5:
                img = torch.flip(img, dims=[2])
            if torch.rand(1) < 0.5:
                img = torch.flip(img, dims=[1])
            k = int(torch.randint(0, 4, (1,)))
            if k:
                img = torch.rot90(img, k, dims=[1, 2])
        if self.classify_threshold is not None:
            y = 1.0 if float(r["pm25"]) > self.classify_threshold else 0.0
        else:
            # log1p target: PM spans 0-466 ug/m3; raw MSE is dominated by smoke days
            y = float(np.log1p(max(float(r["pm25"]), 0.0)))
        return img, torch.from_numpy(self.ctx[i]), torch.tensor(y, dtype=torch.float32)


class FusedNet(nn.Module):
    """ResNet-18 backbone + context fusion (concat at head, or late FiLM)."""
    def __init__(self, backbone: nn.Module, n_ctx: int, fusion: str, feat_dim: int = 512):
        super().__init__()
        self.backbone = backbone          # fc already Identity
        self.n_ctx = n_ctx
        self.fusion = fusion
        self.feat_dim = feat_dim
        if n_ctx == 0:
            self.head = nn.Linear(feat_dim, 1)
        elif fusion == "concat":
            self.ctx_mlp = nn.Sequential(nn.Linear(n_ctx, 64), nn.ReLU(),
                                         nn.Linear(64, 64), nn.ReLU())
            self.head = nn.Sequential(nn.Linear(feat_dim + 64, 128), nn.ReLU(),
                                      nn.Linear(128, 1))
        else:  # film: context produces per-feature scale+shift for the image features
            self.film = nn.Sequential(nn.Linear(n_ctx, 128), nn.ReLU(),
                                      nn.Linear(128, 2 * feat_dim))
            self.head = nn.Sequential(nn.Linear(feat_dim, 128), nn.ReLU(),
                                      nn.Linear(128, 1))

    def forward(self, x, ctx):
        f = self.backbone(x)
        if self.n_ctx == 0:
            return self.head(f)
        if self.fusion == "concat":
            return self.head(torch.cat([f, self.ctx_mlp(ctx)], dim=1))
        gb = self.film(ctx)
        gamma, beta = gb[:, :self.feat_dim], gb[:, self.feat_dim:]
        return self.head(f * (1 + gamma) + beta)


def make_model(in_ch: int = 3) -> nn.Module:
    from torchvision.models import resnet18, ResNet18_Weights
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    if in_ch != 3:
        old = m.conv1.weight.data  # (64, 3, 7, 7)
        conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # inflate: mean of pretrained RGB filters, rescaled to keep activation
        # magnitude comparable to the 3-channel original
        w = old.mean(dim=1, keepdim=True).repeat(1, in_ch, 1, 1) * (3.0 / in_ch)
        conv1.weight.data = w
        m.conv1 = conv1
    m.fc = nn.Linear(m.fc.in_features, 1)
    return m


def make_fused(in_ch: int, n_ctx: int, fusion: str, backbone: str = "resnet18") -> nn.Module:
    import torchvision.models as tvm
    factories = {"resnet18": (tvm.resnet18, tvm.ResNet18_Weights.IMAGENET1K_V1, 512),
                 "resnet34": (tvm.resnet34, tvm.ResNet34_Weights.IMAGENET1K_V1, 512),
                 "resnet50": (tvm.resnet50, tvm.ResNet50_Weights.IMAGENET1K_V2, 2048)}
    fn, weights, feat_dim = factories[backbone]
    bb = fn(weights=weights)
    if in_ch != 3:
        old = bb.conv1.weight.data
        conv1 = nn.Conv2d(in_ch, old.shape[0], kernel_size=7, stride=2, padding=3, bias=False)
        w = torch.zeros(old.shape[0], in_ch, 7, 7)
        w[:, :3] = old  # pretrained RGB filters; extra (diff) channels start at zero
        conv1.weight.data = w
        bb.conv1 = conv1
    bb.fc = nn.Identity()
    return FusedNet(bb, n_ctx, fusion, feat_dim=feat_dim)


def train_one(model, train_dl, val_dl, device, epochs, lr, max_patience=3,
              min_epochs=12, loss_fn=None):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = loss_fn or nn.HuberLoss()
    best_val, best_state, patience = float("inf"), None, 0
    va_hist: list[float] = []
    for ep in range(epochs):
        model.train()
        tr_loss = 0.0
        for x, cx, y in train_dl:
            x, cx, y = x.to(device), cx.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x, cx).squeeze(-1), y)
            loss.backward()
            opt.step()
            tr_loss += float(loss) * len(y)
        sched.step()
        model.eval()
        va_loss, n = 0.0, 0
        with torch.no_grad():
            for x, cx, y in val_dl:
                x, cx, y = x.to(device), cx.to(device), y.to(device)
                va_loss += float(loss_fn(model(x, cx).squeeze(-1), y)) * len(y)
                n += len(y)
        va_loss /= max(n, 1)
        va_hist.append(va_loss)
        # smooth over the last 2 epochs so one noisy epoch can't drive decisions
        va_smooth = sum(va_hist[-2:]) / len(va_hist[-2:])
        print(f"  epoch {ep+1}/{epochs}: train {tr_loss/len(train_dl.dataset):.4f} "
              f"val {va_loss:.4f} (smooth {va_smooth:.4f})")
        if va_smooth < best_val - 1e-4:
            best_val, patience = va_smooth, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            # never stop before min_epochs: unlucky early plateaus were producing
            # undertrained folds (the worst spatial folds all stopped at epoch 7-11)
            if patience >= max_patience and (ep + 1) >= min_epochs:
                print("  early stop")
                break
    if best_state:
        model.load_state_dict(best_state)
    return model


def predict(model, dl, device, tta: bool = False):
    model.eval()
    preds = []
    with torch.no_grad():
        for x, cx, _ in dl:
            x, cx = x.to(device), cx.to(device)
            if not tta:
                preds.append(model(x, cx).squeeze(-1).cpu().numpy())
                continue
            # 8-fold dihedral TTA: 4 rotations x optional mirror, averaged in log space
            outs = []
            for flip in (False, True):
                xf = torch.flip(x, dims=[3]) if flip else x
                for k in range(4):
                    outs.append(model(torch.rot90(xf, k, dims=[2, 3]), cx).squeeze(-1))
            preds.append(torch.stack(outs).mean(0).cpu().numpy())
    return np.expm1(np.concatenate(preds))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="finetune_l1c_scene")
    parser.add_argument("--product", choices=["l2a", "l1c"], default="l1c")
    parser.add_argument("--labels", choices=["weekly", "overpass", "scene", "scenehour",
                                             "rq2daily", "rq2weekly"],
                        default="scene")
    parser.add_argument("--mode", choices=["median", "single", "scene"], default="single")
    parser.add_argument("--bands", choices=["rgb", "all"], default="rgb")
    parser.add_argument("--splits", nargs="+", default=["random", "spatial"])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-epochs", type=int, default=12,
                        help="no early stop before this epoch (undertrained-fold guard)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4,
                        help="DataLoader workers; uses macOS-safe spawn context (0 = in-process)")
    parser.add_argument("--oversample-high", type=float, default=1.0,
                        help=">1 enables WeightedRandomSampler: scenes >35 ug/m3 get this "
                             "weight, 20-35 get half of it, rest 1.0")
    parser.add_argument("--bucket-weights", type=float, nargs=5, default=None,
                        metavar=("W_2.5-6", "W_6-12", "W_12-35", "W_35-55", "W_55+"),
                        help="per-bucket sampling weights (overrides --oversample-high)")
    parser.add_argument("--context", action="store_true",
                        help="fuse 11 context features (met/elev/sun/season/latlon)")
    parser.add_argument("--fusion", choices=["concat", "film"], default="concat",
                        help="context fusion: concat at head, or FiLM modulation of features")
    parser.add_argument("--no-latlon", action="store_true",
                        help="ablation: drop raw lat/lon from the context vector")
    parser.add_argument("--ctx-cols", default=None,
                        help="ablation: comma-separated subset of context columns "
                             "(overrides the default 11; sample set stays identical)")
    parser.add_argument("--backbone", choices=["resnet18", "resnet34", "resnet50"],
                        default="resnet18")
    parser.add_argument("--tta", action="store_true",
                        help="8-fold dihedral test-time augmentation at prediction")
    parser.add_argument("--init-seed", type=int, default=None,
                        help="torch RNG seed for model init only (split unchanged)")
    parser.add_argument("--ref-mode", choices=["clean", "temporal"], default=None,
                        help="6-channel input: image + (image - station reference median)")
    parser.add_argument("--eval-only", default=None, metavar="WEIGHTS",
                        help="skip training: load these weights, rebuild the identical "
                             "split from --seed, and re-score the test set")
    parser.add_argument("--task", choices=["regress", "classify"], default="regress",
                        help="classify = binary clean-vs-elevated head (BCE, pos-weighted)")
    parser.add_argument("--classify-threshold", type=float, default=20.0)
    parser.add_argument("--train-max-pm", type=float, default=None,
                        help="train/val only on scenes with pm25 <= this (clean specialist)")
    parser.add_argument("--train-min-pm", type=float, default=None,
                        help="train/val only on scenes with pm25 > this (specialist regime); "
                             "test set stays complete")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.ctx_cols:
        sel = [c.strip() for c in args.ctx_cols.split(",")]
        bad = [c for c in sel if c not in ALL_CTX_COLS + EXTRA_CTX_COLS]
        if bad:
            raise SystemExit(f"unknown context columns: {bad}")
        CTX_COLS[:] = sel
    if getattr(args, "no_latlon", False):
        for c in ("lat", "lon"):
            CTX_COLS.remove(c)

    cfg = load_config()
    tbl_sfx = "_allscenes" if args.mode == "scene" else ""
    table_path = cfg.path("model_table").with_name(
        f"model_table_{args.product}_{args.labels}{tbl_sfx}.parquet")
    table = pd.read_parquet(table_path)
    # only need keys + label (+ region for the region split); images load from disk
    keep = ["station_id", "week_start", "pm25"] + (["region"] if "region" in table else [])
    table = table[keep].copy()
    if args.context:
        ctx = pd.read_parquet(cfg.path("labels_scenehour").with_name("context_features.parquet"))
        ctx["week_start"] = pd.to_datetime(ctx["key"])
        table["week_start"] = pd.to_datetime(table["week_start"])
        table = table.merge(ctx.drop(columns=["key"]), on=["station_id", "week_start"],
                            how="left")
        n_bad = table[ALL_CTX_COLS].isna().any(axis=1).sum()
        table = table.dropna(subset=ALL_CTX_COLS).reset_index(drop=True)
        extra_sel = [c for c in CTX_COLS if c in EXTRA_CTX_COLS]
        if extra_sel:
            n0 = len(table)
            table = table.dropna(subset=extra_sel).reset_index(drop=True)
            print(f"extra ctx {extra_sel}: dropped {n0 - len(table)} rows missing them")
        print(f"context joined: {len(table)} rows ({n_bad} dropped for missing context)")
    subdir = {"median": "weekly", "single": "single", "scene": "scenes"}[args.mode]
    if args.bands != "rgb":
        subdir = f"{subdir}_{args.bands}"
    patch_root = cfg.path("patches_dir") / args.product / subdir
    if args.ref_mode:
        refs_dir = patch_root.parent / "refs"
        have_ref = {f.name.rsplit("_", 1)[0] for f in refs_dir.glob(f"*_{args.ref_mode}.npy")}
        n0 = len(table)
        table = table[table.station_id.isin(have_ref)].reset_index(drop=True)
        print(f"ref-mode {args.ref_mode}: {len(table)} rows at {table.station_id.nunique()} "
              f"stations with references ({n0 - len(table)} dropped)")
    # all-bands: keep only samples whose 13-band patch exists (download may trail RGB)
    if args.bands != "rgb":
        has = table.apply(lambda r: (patch_root / r["station_id"] /
              f"{pd.Timestamp(r['week_start']).strftime('%Y-%m-%d')}.npy").exists(), axis=1)
        table = table[has].reset_index(drop=True)

    band_stats = None
    if args.bands != "rgb":
        import json as _json
        stats_path = cfg.path("model_table").with_name(f"band_stats_{args.product}_{args.bands}.json")
        if stats_path.exists():
            d = _json.loads(stats_path.read_text())
            band_stats = (np.array(d["mean"], dtype=np.float32),
                          np.array(d["std"], dtype=np.float32))
        else:
            sample = table.sample(min(300, len(table)), random_state=0)
            acc = []
            for r in sample.itertuples():
                wk = pd.Timestamp(r.week_start).strftime("%Y-%m-%d")
                a = np.load(patch_root / r.station_id / f"{wk}.npy").astype(np.float32) / 10000.0
                acc.append(a.reshape(-1, a.shape[-1]))
            allpix = np.concatenate(acc)
            mean, std = allpix.mean(0), allpix.std(0) + 1e-6
            stats_path.write_text(_json.dumps({"mean": mean.tolist(), "std": std.tolist()}))
            band_stats = (mean.astype(np.float32), std.astype(np.float32))
        print(f"band stats over {13} channels ready")
    device = "mps" if torch.backends.mps.is_available() else \
             ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{len(table)} samples, {table['station_id'].nunique()} stations, device={device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run_dir = cfg.path("runs_dir") / args.experiment
    run_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for split_name in args.splits:
        for fold, (tr_idx, te_idx) in enumerate(iter_splits(table, split_name, args.seed)):
            preds_path = run_dir / f"preds_{split_name}_f{fold}.parquet"
            if preds_path.exists():
                prev = pd.read_parquet(preds_path)
                m = compute_metrics(prev["y_true"].to_numpy(), prev["y_pred"].to_numpy(),
                                    prev["station_id"].to_numpy())
                m.update(split=split_name, fold=fold, n_test=len(prev))
                all_rows.append(m)
                print(f"[{split_name} fold {fold}] resumed from existing preds "
                      f"(r2={m['r2']:.3f})")
                continue
            tr = table.iloc[tr_idx]
            te = table.iloc[te_idx]
            if args.train_min_pm is not None:
                before = len(tr)
                tr = tr[tr["pm25"] > args.train_min_pm]
                print(f"  specialist regime: train {before} -> {len(tr)} scenes "
                      f"(pm > {args.train_min_pm})")
            if args.train_max_pm is not None:
                before = len(tr)
                tr = tr[tr["pm25"] <= args.train_max_pm]
                print(f"  clean regime: train {before} -> {len(tr)} scenes "
                      f"(pm <= {args.train_max_pm})")
            # station-aware val carve-out from train (for early stopping)
            val_stations = tr["station_id"].drop_duplicates().sample(
                frac=0.15, random_state=args.seed)
            va = tr[tr["station_id"].isin(val_stations)]
            tr2 = tr[~tr["station_id"].isin(val_stations)]
            print(f"[{split_name} fold {fold}] train {len(tr2)} val {len(va)} test {len(te)}")

            # fork-context workers deadlock with MPS on macOS; spawn context is
            # the safe way to parallelize loading (matters for 13-band patches,
            # where single-threaded decode starves the GPU)
            loader_kw = dict(batch_size=args.batch_size, num_workers=args.workers)
            if args.workers > 0:
                loader_kw.update(multiprocessing_context="spawn", persistent_workers=True)
            ctx_stats = None
            if args.context:
                cm = tr[CTX_COLS].mean().to_numpy(dtype="float32")
                cs = tr[CTX_COLS].std().to_numpy(dtype="float32") + 1e-6
                ctx_stats = (cm, cs)

            def dl(rows, train):
                ds = PatchDataset(rows, patch_root, cfg.embeddings["rgb_gain"], train,
                                  band_stats=band_stats,
                                  classify_threshold=(args.classify_threshold
                                                      if args.task == "classify" else None),
                                  ctx_stats=ctx_stats, ref_mode=args.ref_mode)
                if train and (args.oversample_high > 1.0 or args.bucket_weights):
                    import numpy as _np
                    from torch.utils.data import WeightedRandomSampler
                    pm = rows["pm25"].to_numpy()
                    if args.bucket_weights:
                        edges = [-1, 6, 12, 35, 55, 1e9]
                        w = _np.ones(len(pm))
                        for i in range(5):
                            w[(pm > edges[i]) & (pm <= edges[i+1])] = args.bucket_weights[i]
                    else:
                        w = _np.ones(len(pm))
                        w[pm > 20] = args.oversample_high / 2.0
                        w[pm > 35] = args.oversample_high
                    sampler = WeightedRandomSampler(
                        torch.as_tensor(w, dtype=torch.double), num_samples=len(ds),
                        replacement=True)
                    return DataLoader(ds, sampler=sampler, **loader_kw)
                return DataLoader(ds, shuffle=train, **loader_kw)
            in_ch = 3 if band_stats is None else len(band_stats[0])
            if args.ref_mode:
                in_ch += 3
            n_ctx = len(CTX_COLS) if args.context else 0
            if args.init_seed is not None:
                torch.manual_seed(args.init_seed)
            model = make_fused(in_ch, n_ctx, args.fusion, args.backbone).to(device)
            if args.eval_only:
                ck = torch.load(args.eval_only, map_location=device)
                model.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
                print(f"  eval-only: loaded {args.eval_only}")
                y_pred = predict(model, dl(te, False), device, tta=args.tta)
                m = compute_metrics(te["pm25"].to_numpy(), y_pred, te["station_id"].to_numpy())
                m.update(split=split_name, fold=fold, n_test=len(te))
                all_rows.append(m)
                pd.DataFrame({"station_id": te["station_id"], "week_start": te["week_start"],
                              "y_true": te["pm25"], "y_pred": y_pred}).to_parquet(
                    run_dir / f"preds_{split_name}_f{fold}.parquet", index=False)
                print(f"  -> r2={m['r2']:.3f} rmse={m['rmse']:.2f} "
                      f"between={m['between_station_r2']:.3f} within={m['within_station_r2']:.3f}")
                continue
            loss_fn = None
            if args.task == "classify":
                pos = float((tr2["pm25"] > args.classify_threshold).sum())
                neg = float(len(tr2) - pos)
                loss_fn = nn.BCEWithLogitsLoss(
                    pos_weight=torch.tensor(neg / max(pos, 1.0)).to(device))
                print(f"  classifier: {int(pos)} positive / {int(neg)} negative "
                      f"(pos_weight {neg/max(pos,1.0):.1f})")
            model = train_one(model, dl(tr2, True), dl(va, False), device,
                              args.epochs, args.lr, args.patience, args.min_epochs,
                              loss_fn=loss_fn)
            if split_name in ("final", "holdout"):
                out_path = run_dir / f"model_{split_name}_s{args.seed}.pt"
                torch.save({"state_dict": model.state_dict(),
                            "recipe": vars(args)}, out_path)
                print(f"saved model weights -> {out_path}")
                if split_name == "final":
                    continue
            if args.task == "classify":
                import torch as _t
                model.eval(); probs = []
                with _t.no_grad():
                    for x, cx, _ in dl(te, False):
                        probs.append(_t.sigmoid(model(x.to(device), cx.to(device)).squeeze(-1)).cpu().numpy())
                y_pred = np.concatenate(probs)
                from thesis.models.metrics import roc_auc
                auc = roc_auc((te["pm25"] > args.classify_threshold).to_numpy(), y_pred)
                print(f"  -> classifier AUC = {auc:.4f}")
                pd.DataFrame({"station_id": te["station_id"], "week_start": te["week_start"],
                              "y_true": te["pm25"], "y_pred": y_pred}).to_parquet(
                    run_dir / f"preds_{split_name}_f{fold}.parquet", index=False)
                all_rows.append({"split": split_name, "fold": fold, "auc": auc,
                                 "r2": float("nan"), "rmse": float("nan"), "mae": float("nan"),
                                 "between_station_r2": float("nan"),
                                 "within_station_r2": float("nan"), "n_test": len(te)})
                continue
            y_pred = predict(model, dl(te, False), device)
            m = compute_metrics(te["pm25"].to_numpy(), y_pred, te["station_id"].to_numpy())
            m.update(split=split_name, fold=fold, n_test=len(te))
            all_rows.append(m)
            print(f"  -> r2={m['r2']:.3f} rmse={m['rmse']:.2f} "
                  f"between={m['between_station_r2']:.3f} within={m['within_station_r2']:.3f}")
            pd.DataFrame({"station_id": te["station_id"], "week_start": te["week_start"],
                          "y_true": te["pm25"], "y_pred": y_pred}).to_parquet(
                run_dir / f"preds_{split_name}_f{fold}.parquet", index=False)

    if not all_rows:
        print("no evaluation rows (final mode) - weights saved, skipping summary")
        return 0
    res = pd.DataFrame(all_rows)
    summary = res.groupby("split")[["r2", "rmse", "mae", "between_station_r2",
                                    "within_station_r2"]].mean().round(3)
    print("\n== summary ==")
    print(summary.to_string())
    res.to_parquet(run_dir / "results.parquet", index=False)
    (run_dir / "summary.json").write_text(summary.to_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
