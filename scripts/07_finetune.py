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


class PatchDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, patch_root: Path, gain: float, train: bool,
                 band_stats: tuple | None = None):
        self.rows = rows.reset_index(drop=True)
        self.root = patch_root
        self.gain = gain
        self.train = train
        self.band_stats = band_stats  # (mean[C], std[C]) in reflectance units, or None for RGB path

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
        if self.train:
            if torch.rand(1) < 0.5:
                img = torch.flip(img, dims=[2])
            if torch.rand(1) < 0.5:
                img = torch.flip(img, dims=[1])
            k = int(torch.randint(0, 4, (1,)))
            if k:
                img = torch.rot90(img, k, dims=[1, 2])
        # log1p target: PM spans 0-466 ug/m3; raw-scale MSE is dominated by smoke days
        y = float(np.log1p(max(float(r["pm25"]), 0.0)))
        return img, torch.tensor(y, dtype=torch.float32)


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


def train_one(model, train_dl, val_dl, device, epochs, lr, max_patience=3,
              min_epochs=12):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.HuberLoss()
    best_val, best_state, patience = float("inf"), None, 0
    va_hist: list[float] = []
    for ep in range(epochs):
        model.train()
        tr_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x).squeeze(-1), y)
            loss.backward()
            opt.step()
            tr_loss += float(loss) * len(y)
        sched.step()
        model.eval()
        va_loss, n = 0.0, 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                va_loss += float(loss_fn(model(x).squeeze(-1), y)) * len(y)
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


def predict(model, dl, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for x, _ in dl:
            preds.append(model(x.to(device)).squeeze(-1).cpu().numpy())
    return np.expm1(np.concatenate(preds))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="finetune_l1c_scene")
    parser.add_argument("--product", choices=["l2a", "l1c"], default="l1c")
    parser.add_argument("--labels", choices=["weekly", "overpass", "scene", "scenehour"],
                        default="scene")
    parser.add_argument("--mode", choices=["median", "single"], default="single")
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
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config()
    table_path = cfg.path("model_table").with_name(
        f"model_table_{args.product}_{args.labels}.parquet")
    table = pd.read_parquet(table_path)
    # only need keys + label (+ region for the region split); images load from disk
    keep = ["station_id", "week_start", "pm25"] + (["region"] if "region" in table else [])
    table = table[keep].copy()
    subdir = "weekly" if args.mode == "median" else "single"
    if args.bands != "rgb":
        subdir = f"{subdir}_{args.bands}"
    patch_root = cfg.path("patches_dir") / args.product / subdir
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
            dl = lambda rows, train: DataLoader(
                PatchDataset(rows, patch_root, cfg.embeddings["rgb_gain"], train,
                             band_stats=band_stats),
                shuffle=train, **loader_kw)
            in_ch = 3 if band_stats is None else len(band_stats[0])
            model = make_model(in_ch).to(device)
            model = train_one(model, dl(tr2, True), dl(va, False), device,
                              args.epochs, args.lr, args.patience, args.min_epochs)
            if split_name == "final":
                out_path = run_dir / f"model_final_s{args.seed}.pt"
                torch.save({"state_dict": model.state_dict(),
                            "recipe": vars(args)}, out_path)
                print(f"saved final model weights -> {out_path}")
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
