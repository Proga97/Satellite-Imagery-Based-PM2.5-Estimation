"""Context-only baseline: the eleven context features with NO image, on the same scenes and
station splits as the fused models (holdout seed 0 = split A, seed 1 = split B).

Answers the committee question "what does the context give on its own?". Two learners on
the same log1p target: a small MLP (same target/loss family as the CNN) and a nearest-neighbour regressor.
Predictions are written in the same format as 07_finetune.py so 11_figures.py can read them.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("ft", ROOT / "scripts/07_finetune.py")
ft = importlib.util.module_from_spec(spec); spec.loader.exec_module(ft)
from thesis.config import load_config  # noqa: E402

cfg = load_config()
CTX = ft.ALL_CTX_COLS


def load_table():
    table = pd.read_parquet(cfg.path("model_table").with_name("model_table_l1c_scenehour_allscenes.parquet"))
    table = table[["station_id", "week_start", "pm25", "region"]].copy()
    ctx = pd.read_parquet(cfg.path("labels_scenehour").with_name("context_features.parquet"))
    ctx["week_start"] = pd.to_datetime(ctx["key"]); table["week_start"] = pd.to_datetime(table["week_start"])
    table = table.merge(ctx.drop(columns=["key"]), on=["station_id", "week_start"], how="left")
    return table.dropna(subset=CTX).reset_index(drop=True)


def r2(y, p): return 1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def fit_mlp(Xtr, ytr, Xva, yva, seed):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(Xtr.shape[1], 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.HuberLoss()
    Xtr_t, ytr_t = torch.tensor(Xtr), torch.tensor(ytr); Xva_t, yva_t = torch.tensor(Xva), torch.tensor(yva)
    best, best_state, bad = 1e9, None, 0
    for epoch in range(200):
        net.train(); perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(perm), 256):
            idx = perm[i:i + 256]; opt.zero_grad()
            loss = loss_fn(net(Xtr_t[idx]).squeeze(1), ytr_t[idx]); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            v = loss_fn(net(Xva_t).squeeze(1), yva_t).item()
        if v < best - 1e-4:
            best, bad = v, 0; best_state = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 15: break
    net.load_state_dict(best_state); net.eval()
    return lambda X: net(torch.tensor(X)).squeeze(1).detach().numpy()


def fit_knn(Xtr, ytr, Xva, yva, seed, k=50):
    """Nearest-neighbour regression in standardized context space (no scipy/sklearn: both
    are broken on this machine). A nonparametric foil for the MLP."""
    Xtr_t, ytr_t = torch.tensor(Xtr), torch.tensor(ytr)
    def predict(X):
        out = []
        Xt = torch.tensor(X)
        for i in range(0, len(Xt), 2048):
            d = torch.cdist(Xt[i:i + 2048], Xtr_t); idx = d.topk(k, largest=False).indices
            out.append(ytr_t[idx].mean(1))
        return torch.cat(out).numpy()
    return predict


def main():
    table = load_table(); print(f"{len(table)} scenes, {table.station_id.nunique()} stations, features: {CTX}")
    out = {}
    for name, seed in [("A", 0), ("B", 1)]:
        tr_idx, te_idx = next(ft.iter_splits(table, "holdout", seed))
        tr, te = table.iloc[tr_idx], table.iloc[te_idx]
        val_st = tr["station_id"].drop_duplicates().sample(frac=0.15, random_state=seed)
        va, tr2 = tr[tr.station_id.isin(val_st)], tr[~tr.station_id.isin(val_st)]
        cm, cs = tr2[CTX].mean().to_numpy("float32"), tr2[CTX].std().to_numpy("float32") + 1e-6
        X = lambda d: ((d[CTX].to_numpy("float32") - cm) / cs)
        y = lambda d: np.log1p(d.pm25.to_numpy("float32"))
        print(f"[split {name}] train {len(tr2)} val {len(va)} test {len(te)} ({te.station_id.nunique()} stations)")
        for learner, fit in [("mlp", fit_mlp), ("knn", fit_knn)]:
            pred = fit(X(tr2), y(tr2), X(va), y(va), seed)
            yp = np.expm1(pred(X(te))).clip(min=0); yt = te.pm25.to_numpy()
            d = ROOT / f"data/runs/ctxonly_{learner}_{name}"; d.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"station_id": te.station_id.values, "week_start": te.week_start.dt.strftime("%Y-%m-%d").values,
                          "y_true": yt, "y_pred": yp}).to_parquet(d / "preds_holdout_f0.parquet", index=False)
            g = pd.DataFrame({"s": te.station_id.values, "t": yt, "p": yp}).groupby("s")[["t", "p"]].mean()
            res = dict(r2=round(float(r2(yt, yp)), 3), mae=round(float(np.abs(yp - yt).mean()), 2),
                       r=round(float(np.corrcoef(yt, yp)[0, 1]), 3),
                       station_avg_corr=round(float(np.corrcoef(g.t, g.p)[0, 1]), 2),
                       station_avg_rmse=round(float(np.sqrt(((g.p - g.t) ** 2).mean())), 2))
            out[f"{learner}_{name}"] = res; print(f"  {learner:5} split {name}: {res}")
            json.dump(res, open(d / "summary.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
