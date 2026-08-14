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
from thesis.models.splits import random_split, spatial_folds


def iter_splits(table: pd.DataFrame, name: str, seed: int):
    if name == "random":
        yield from random_split(table, seed=seed)
    elif name == "spatial":
        yield from spatial_folds(table)
    else:
        raise ValueError(f"unsupported split for finetune: {name}")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class PatchDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, patch_root: Path, gain: float, train: bool):
        self.rows = rows.reset_index(drop=True)
        self.root = patch_root
        self.gain = gain
        self.train = train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows.iloc[i]
        wk = pd.Timestamp(r["week_start"]).strftime("%Y-%m-%d")
        arr = np.load(self.root / r["station_id"] / f"{wk}.npy").astype(np.float32)
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
        # log1p target: PM spans 1-466 ug/m3; raw-scale MSE is dominated by smoke days
        y = float(np.log1p(r["pm25"]))
        return img, torch.tensor(y, dtype=torch.float32)


def make_model() -> nn.Module:
    from torchvision.models import resnet18, ResNet18_Weights
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, 1)
    return m


def train_one(model, train_dl, val_dl, device, epochs, lr):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.HuberLoss()
    best_val, best_state, patience = float("inf"), None, 0
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
        print(f"  epoch {ep+1}/{epochs}: train {tr_loss/len(train_dl.dataset):.4f} val {va_loss:.4f}")
        if va_loss < best_val - 1e-4:
            best_val, patience = va_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 3:
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
    parser.add_argument("--labels", choices=["weekly", "overpass", "scene"], default="scene")
    parser.add_argument("--mode", choices=["median", "single"], default="single")
    parser.add_argument("--splits", nargs="+", default=["random", "spatial"])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config()
    table_path = cfg.path("model_table").with_name(
        f"model_table_{args.product}_{args.labels}.parquet")
    table = pd.read_parquet(table_path)
    # only need keys + label; images are loaded from disk
    table = table[["station_id", "week_start", "pm25"]].copy()
    subdir = "weekly" if args.mode == "median" else "single"
    patch_root = cfg.path("patches_dir") / args.product / subdir
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
            tr = table.iloc[tr_idx]
            te = table.iloc[te_idx]
            # station-aware val carve-out from train (for early stopping)
            val_stations = tr["station_id"].drop_duplicates().sample(
                frac=0.15, random_state=args.seed)
            va = tr[tr["station_id"].isin(val_stations)]
            tr2 = tr[~tr["station_id"].isin(val_stations)]
            print(f"[{split_name} fold {fold}] train {len(tr2)} val {len(va)} test {len(te)}")

            dl = lambda rows, train: DataLoader(
                PatchDataset(rows, patch_root, cfg.embeddings["rgb_gain"], train),
                batch_size=args.batch_size, shuffle=train, num_workers=4,
                persistent_workers=False)
            model = make_model().to(device)
            model = train_one(model, dl(tr2, True), dl(va, False), device,
                              args.epochs, args.lr)
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
