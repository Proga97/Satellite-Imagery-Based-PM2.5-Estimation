#!/usr/bin/env python
"""Regenerate every thesis figure into docs/figures/ from the run artifacts.

Reads: prediction parquets in data/runs/, label/station parquets in data/interim/,
the allscenes model table, patches for the scene pair, and training logs.
Numbers hardcoded from EXPERIMENT_LOG.md are marked with the section they cite.
Usage: python scripts/11_figures.py [figN ...]   (no args = all)
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({"figure.dpi": 150, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
FIG = Path("docs/figures"); FIG.mkdir(parents=True, exist_ok=True)
BLUE, LTBLUE, GRAY, RED, GREEN = "#2e86c1", "#7fb3d5", "#b0b0b0", "#c0392b", "#16a085"
STATE_COLORS = [BLUE, GREEN, "#e67e22", "#8e44ad", RED]
KEYS = ["station_id", "week_start"]
STATE_OF = {"06": "CA", "48": "TX", "36": "NY", "53": "WA", "17": "IL"}
ORDER = ["CA", "WA", "TX", "IL", "NY"]
# certified 5-member honest blend (§3w): TTA'd trio + seed-2 + temporal-reference
SPLIT_A = ["tta_film_s0", "tta_nolatlon_s0", "tta_geotime_s0",
           "film_full_seed2", "film_reftemporal_s0"]
SPLIT_B = ["tta_film_s1", "tta_nolatlon_s1", "tta_geotime_s1",
           "film_full_seed2_B", "film_reftemporal_B"]
BUCK = [(2.5, 6, "2.5–6"), (6, 12, "6–12"), (12, 35, "12–35"), (35, 55, "35–55"), (55, 1e9, "55+")]


def preds(run):
    return (pd.read_parquet(f"data/runs/{run}/preds_holdout_f0.parquet")
            .assign(week_start=lambda d: pd.to_datetime(d.week_start)))


def blend(runs):
    dfs = [preds(r) for r in runs]
    b = dfs[0][KEYS + ["y_true"]].copy()
    for i, d in enumerate(dfs):
        b[f"m{i}"] = b.set_index(KEYS).index.map(d.set_index(KEYS).y_pred)
    b = b.dropna().reset_index(drop=True)
    b["y_pred"] = b[[f"m{i}" for i in range(len(dfs))]].mean(axis=1)
    return b


def save(fig, name):
    fig.tight_layout(); fig.savefig(FIG / name, bbox_inches="tight"); plt.close(fig)
    print(name)


def scenes_table():
    t = pd.read_parquet("data/processed/model_table_l1c_scenehour_allscenes.parquet")[
        ["station_id", "week_start", "pm25"]]
    t["date"] = pd.to_datetime(t.week_start)
    t["state"] = t.station_id.str[:2].map(STATE_OF)
    return t


def fig1():  # story arc (§2 ladder + §3p/§3t)
    stages = [("Proposal config\n(L2A weekly composite,\nfrozen CNN)", 0.00),
              ("L1C single scenes\n+ label sync", 0.19),
              ("+ wildfire era,\nfine-tuned end-to-end\n(CA only)", 0.39),
              ("5 states,\nrare-event weighting\n(damped)", 0.31),
              ("Image-only\nensemble ×3\n(certified §3p)", 0.362),
              ("Triple-FiLM\ncontext blend\n(certified §3t)", 0.437),
              ("5-member blend\n+TTA +references\n(certified §3w)", 0.435)]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    vals = [v for _, v in stages]
    ax.bar(range(len(stages)), vals, color=[GRAY]*4 + [LTBLUE, LTBLUE, BLUE], width=0.62)
    for x, v in enumerate(vals):
        ax.text(x, v + 0.008, f"{v:.2f}", ha="center", fontweight="bold")
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels([s for s, _ in stages], fontsize=7.5)
    ax.set_ylabel("R² at never-seen stations")
    ax.set_title("From zero signal to a certified instrument — the project arc")
    save(fig, "fig1_story_arc.png")


def fig2():
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), sharex=True, sharey=True)
    for ax, b, ttl, r2 in [(axes[0], blend(SPLIT_A), "Split A (seed 0)", 0.435),
                           (axes[1], blend(SPLIT_B), "Split B (seed 1)", 0.435)]:
        ax.scatter(b.y_true, b.y_pred, s=4, alpha=0.18, color=BLUE, edgecolors="none")
        lim = [2, 300]
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.axvline(35, color=RED, lw=0.7, ls=":"); ax.axhline(35, color=RED, lw=0.7, ls=":")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_title(f"{ttl} — R²={r2:.3f}, n={len(b):,}")
        ax.set_xlabel("EPA measured PM2.5 (µg/m³)")
    axes[0].set_ylabel("Predicted PM2.5 (µg/m³)")
    fig.suptitle("Champion (5-member FiLM blend, §3w) at held-out stations", y=1.0)
    save(fig, "fig2_champion_scatter.png")


def fig3():  # §3q-3s ablation
    rows = [("Image-only (damped)", 0.307, 0.103), ("Concat fusion (11 feat)", 0.304, -0.373),
            ("FiLM full ctx (11 feat)", 0.377, 0.146), ("FiLM physics-only (9)", 0.304, 0.483),
            ("FiLM geo+time (4)", 0.380, 0.405), ("Triple-FiLM blend", 0.406, 0.513)]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, idx, ttl in [(axes[0], 1, "Overall R²"),
                         (axes[1], 2, "Between-station R² (ranking places)")]:
        v = [r[idx] for r in rows]
        colors = [RED if x < 0 else (BLUE if i == len(rows) - 1 else LTBLUE)
                  for i, x in enumerate(v)]
        ax.barh(range(len(rows)), v, color=colors, height=0.6)
        ax.axvline(0, color="k", lw=0.8)
        for i, x in enumerate(v):
            ax.text(x + (0.012 if x >= 0 else -0.012), i,
                    f"{x:+.2f}" if idx == 2 else f"{x:.2f}",
                    va="center", ha="left" if x >= 0 else "right", fontsize=8)
        ax.set_title(ttl, fontsize=10)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
        ax.invert_yaxis()
    fig.suptitle("Context-fusion ablation (same split): how you fuse matters more than what you fuse",
                 y=1.02)
    save(fig, "fig3_ablation_ladder.png")


def fig4():
    fig, ax = plt.subplots(figsize=(7, 3.8))
    w = 0.38
    for off, b, lab, col in [(-w/2, blend(SPLIT_A), "Split A", LTBLUE),
                             (w/2, blend(SPLIT_B), "Split B", BLUE)]:
        maes, biases = [], []
        for lo, hi, name in BUCK:
            s = (b[(b.y_true >= lo) & (b.y_true <= hi)] if name == "2.5–6"
                 else b[(b.y_true > lo) & (b.y_true <= hi)])
            e = s.y_pred - s.y_true
            maes.append(e.abs().mean()); biases.append(e.mean())
        xs = np.arange(len(BUCK)) + off
        ax.bar(xs, maes, width=w, color=col, label=f"{lab} MAE")
        for x, bias, mae in zip(xs, biases, maes):
            ax.text(x, mae + 1.2, f"{bias:+.0f}", ha="center", fontsize=7.5, color="#555")
    ax.set_xticks(range(len(BUCK))); ax.set_xticklabels([n for _, _, n in BUCK])
    ax.set_xlabel("True PM2.5 bucket (µg/m³)"); ax.set_ylabel("MAE (µg/m³)")
    ax.set_title("Champion error by pollution level (numbers above bars = mean bias)")
    ax.legend(fontsize=8)
    save(fig, "fig4_bucket_profile.png")


def fig5():  # §3u
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    arms = ["Hourly ±1h", "Daily mean", "Weekly mean"]
    for ax, v, ttl in [(axes[0], [0.377, 0.377, 0.321], "Per-scene: predicting its own label"),
                       (axes[1], [0.262, 0.325, 0.383], "Weekly product: scene-averaged vs weekly truth")]:
        best = max(v)
        ax.bar(arms, v, color=[BLUE if x == best else LTBLUE for x in v], width=0.55)
        for i, x in enumerate(v):
            ax.text(i, x + 0.006, f"{x:.3f}", ha="center", fontweight="bold", fontsize=9)
        ax.set_title(ttl, fontsize=9.5); ax.set_ylabel("R²"); ax.set_ylim(0, 0.45)
    fig.suptitle("RQ2 — label aggregation: sync matters to daily, then match the label to the product",
                 y=1.02)
    save(fig, "fig5_rq2.png")


def fig6():
    sid = "06-027-0002"
    lab = pd.read_parquet("data/interim/labels_scenehour.parquet")
    lab["key"] = lab["scene_date"].astype(str).str[:10]
    man = pd.read_parquet("data/patches/manifest.parquet")
    man = man[(man.status == "ok") & (man.bands == "rgb") & (man["mode"] == "scene")
              & (man.station_id == sid)].copy()
    man["key"] = man.scene_date.astype(str).str[:10]
    mm = man.merge(lab[lab.station_id == sid][["key", "pm25"]].drop_duplicates("key"), on="key")
    mm["path"] = mm.key.map(lambda k: f"data/patches/l1c/scenes/{sid}/{k}.npy")
    mm = mm[mm.path.map(lambda p: Path(p).exists())]
    smoke = mm[mm.pm25 > 80].sort_values("pm25").iloc[-1]
    cl = mm[mm.pm25.between(4, 8) & mm.key.str[5:7].isin(["07", "08", "09", "10"])]
    clean = (cl if len(cl) else mm[mm.pm25.between(4, 8)]).iloc[0]

    def load(row):
        a = np.load(row.path).astype(np.float32) / 10000.0
        return np.clip(a[..., ::-1] * 3.0, 0, 1)  # B4,B3,B2 -> RGB, eyeball gain 3

    fig, axes = plt.subplots(1, 2, figsize=(8, 4.2))
    for ax, row, ttl in [(axes[0], clean, f"Clean day — {clean.key}\nEPA: {clean.pm25:.1f} µg/m³"),
                         (axes[1], smoke, f"Wildfire smoke — {smoke.key}\nEPA: {smoke.pm25:.1f} µg/m³")]:
        ax.imshow(load(row)); ax.set_title(ttl, fontsize=10); ax.axis("off")
    fig.suptitle(f"Same station ({sid}, San Joaquin Valley CA) — 2.24 km patch", y=0.98)
    save(fig, "fig6_scene_pair.png")


def used_stations():
    st = pd.read_parquet("data/interim/stations.parquet")
    tbl = pd.read_parquet("data/processed/model_table_l1c_scenehour_allscenes.parquet")
    return st[st.station_id.isin(set(tbl.station_id))]


def fig7():
    st = used_stations()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(st.lon, st.lat, s=14, color=BLUE, alpha=0.75, edgecolors="none")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title(f"EPA PM2.5 reference stations used ({len(st)} stations, 5 states, 2020–2025)")
    ax.set_aspect(1.25)
    for name, (x, y) in {"CA": (-120.5, 36.5), "WA": (-121.0, 47.7), "TX": (-98.5, 30.5),
                         "IL": (-89.0, 40.5), "NY": (-75.5, 43.0)}.items():
        ax.annotate(name, (x, y), fontsize=12, fontweight="bold", color="#555")
    save(fig, "fig7_station_map.png")


def fig8():
    tbl = scenes_table()
    fig, ax = plt.subplots(figsize=(8, 3.8))
    bins = np.logspace(np.log10(2.5), np.log10(500), 60)
    ax.hist(tbl.pm25, bins=bins, color=BLUE, alpha=0.85)
    ax.set_xscale("log"); ax.set_yscale("log")
    for x in (6, 12, 35, 55):
        ax.axvline(x, color=RED, lw=0.8, ls=":")
    ax.set_xlabel("PM2.5 at overpass ±1h (µg/m³, log scale)"); ax.set_ylabel("scenes (log)")
    ax.set_title(f"Label distribution — {len(tbl):,} scenes; medians hide a 100× dynamic range")
    save(fig, "fig8_label_distribution.png")


def fig9():
    tbl = scenes_table()
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    data = [tbl[tbl.state == s].pm25 for s in ORDER]
    bp = ax.boxplot(data, tick_labels=[f"{s}\n(n={len(d):,})" for s, d in zip(ORDER, data)],
                    showfliers=False, patch_artist=True, medianprops=dict(color="k"))
    for p in bp["boxes"]:
        p.set_facecolor(LTBLUE)
    for x, d in enumerate(data, start=1):
        ax.text(x, d.quantile(0.75) + 2.2, f"p99={d.quantile(0.99):.0f}", ha="center",
                fontsize=8, color="#555")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.set_title("PM2.5 by state (boxes: IQR; p99 shows the wildfire tail)")
    save(fig, "fig9_state_distribution.png")


def fig10():
    tbl = scenes_table()
    tbl["month"] = tbl.date.dt.month
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for s, col in zip(ORDER, STATE_COLORS):
        mm = tbl[tbl.state == s].groupby("month").pm25.mean()
        ax.plot(mm.index, mm.values, marker="o", ms=3.5, label=s, color=col, lw=1.4)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.set_ylabel("mean PM2.5 (µg/m³)"); ax.set_xlabel("month")
    ax.set_title("Seasonal cycle by state — the pattern doy_sin/doy_cos lets the model learn")
    ax.legend(fontsize=8, ncol=5)
    save(fig, "fig10_seasonal_cycle.png")


def fig11():
    tbl = scenes_table()
    tbl["year"] = tbl.date.dt.year
    piv = tbl.pivot_table(index="year", columns="state", values="pm25", aggfunc="size").fillna(0)[ORDER]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    bottom = np.zeros(len(piv))
    for s, col in zip(ORDER, STATE_COLORS):
        ax.bar(piv.index, piv[s], bottom=bottom, label=s, color=col, width=0.65)
        bottom += piv[s].values
    ax.set_ylabel("scenes")
    ax.set_title("Scenes per year by state (every usable satellite pass, 2020–2025)")
    ax.legend(fontsize=8, ncol=5)
    save(fig, "fig11_scenes_per_year.png")


def fig12():
    sk = pd.read_parquet("data/interim/scene_keep.parquet")
    r = sk.reason.value_counts()
    steps = [("All downloaded\ncurrent-era scenes", len(sk), GRAY),
             ("− PM < 2.5\n(unreliable low range)", -int(r["pm_below_2.5"]), RED),
             ("− cloudy scene,\nlow label", -int(r["cloud_low_label"]), RED),
             ("− single-hour\nlabel", -int(r["single_hour_label"]), RED),
             ("− tile-edge\nzeros", -int(r["tile_edge_zeros"]), RED),
             ("Kept for\ntraining", int(sk.keep.sum()), BLUE)]
    fig, ax = plt.subplots(figsize=(8.5, 4))
    run = 0
    for i, (name, v, col) in enumerate(steps):
        if i == 0:
            ax.bar(i, v, color=col, width=0.6); run = v
        elif i == len(steps) - 1:
            ax.bar(i, v, color=col, width=0.6)
        else:
            ax.bar(i, -v, bottom=run + v, color=col, width=0.6); run += v
        ax.text(i, (run if 0 < i < len(steps) - 1 else v) + 900, f"{abs(v):,}",
                ha="center", fontsize=8.5, fontweight="bold")
    ax.set_xticks(range(len(steps))); ax.set_xticklabels([s for s, _, _ in steps], fontsize=8)
    ax.set_ylabel("scenes"); ax.set_title("Data cleaning waterfall (rules R1–R4, §1 of the experiment log)")
    save(fig, "fig12_cleaning_waterfall.png")


def fig13():  # §2 ladder rows 1-10
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    ax = axes[0]
    mat = np.array([[0.00, 0.04], [0.01, 0.00]])
    ax.imshow(mat, cmap="RdYlGn", vmin=-0.05, vmax=0.45)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["weekly label", "day-synced label"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["L2A\n(corrected)", "L1C\n(raw TOA)"])
    ax.set_title("2×2 with frozen features + composites:\nall dead (spatial R²)", fontsize=9)
    ax = axes[1]
    levers = [("weekly median\ncomposites", 0.00), ("single scenes,\nscene-date label", 0.15),
              ("+ hour sync", 0.17), ("+ fine-tuned\nend-to-end", 0.39)]
    v = [x for _, x in levers]
    ax.bar(range(4), v, color=[GRAY, LTBLUE, LTBLUE, BLUE], width=0.6)
    for i, x in enumerate(v):
        ax.text(i, x + 0.008, f"{x:.2f}", ha="center", fontweight="bold", fontsize=9)
    ax.set_xticks(range(4)); ax.set_xticklabels([n for n, _ in levers], fontsize=7.5)
    ax.set_title("The two levers that unlocked signal\n(California-era runs)", fontsize=9)
    ax.set_ylabel("spatial R²")
    save(fig, "fig13_dead_matrix_levers.png")


def fig14():
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 5)

    def box(x, y, w, h, text, fc, fs=8.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                    fc=fc, ec="#444", lw=0.8))
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=13, color="#444", lw=1.1))
    box(0.2, 3.2, 1.7, 1.3, "Sentinel-2 L1C\npatch 224×224×3\n(raw TOA)", "#dbe9f5")
    box(2.6, 3.35, 1.9, 1.0, "ResNet-18\nbackbone", LTBLUE)
    box(5.1, 3.35, 1.5, 1.0, "image features\nf ∈ ℝ⁵¹²", "#dbe9f5")
    box(0.2, 0.6, 1.7, 1.5, "context vector\nweather · elevation\nsun · season · lat/lon", "#d9efe6")
    box(2.6, 0.85, 1.9, 1.0, "MLP 64→64", GREEN)
    box(5.1, 0.85, 1.5, 1.0, "γ, β ∈ ℝ⁵¹²", "#d9efe6")
    box(7.0, 2.0, 1.4, 1.1, "FiLM\nf·(1+γ)+β", "#f6e6c8")
    box(8.8, 2.05, 1.1, 1.0, "head\n→ PM2.5", BLUE)
    arrow(1.9, 3.85, 2.6, 3.85); arrow(4.5, 3.85, 5.1, 3.85); arrow(6.6, 3.7, 7.3, 3.1)
    arrow(1.9, 1.35, 2.6, 1.35); arrow(4.5, 1.35, 5.1, 1.35); arrow(6.6, 1.5, 7.3, 2.1)
    arrow(8.4, 2.55, 8.8, 2.55)
    ax.text(7.7, 0.7, "context can only MODULATE how the image is read —\n"
            "it has no direct path to the output (vs. concat, which does\n"
            "and collapses at unseen stations)", fontsize=8, ha="center", color="#555")
    ax.set_title("FiLM context fusion — the champion architecture (×3 context variants, blended)")
    save(fig, "fig14_architecture.png")


def fig15():
    st = used_stations()
    test0 = set(preds("fused_film").station_id)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    tr = st[~st.station_id.isin(test0)]; te = st[st.station_id.isin(test0)]
    ax.scatter(tr.lon, tr.lat, s=13, color=GRAY, alpha=0.7,
               label=f"train+val pool ({len(tr)})", edgecolors="none")
    ax.scatter(te.lon, te.lat, s=26, color=RED, alpha=0.9,
               label=f"held-out test ({len(te)}, never seen)", edgecolors="none")
    ax.set_aspect(1.25); ax.legend(fontsize=9, loc="lower left")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("Evaluation protocol: 20% of stations per state held out entirely (split A shown)")
    save(fig, "fig15_split_protocol.png")


def fig16():
    txt = open("data/runs/reval_film.log").read()
    segs = txt.split("context joined")[1:]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for seg, name, col in zip(segs, ["full context", "physics-only", "geo+time"],
                              [BLUE, GREEN, "#e67e22"]):
        ep = re.findall(r"epoch (\d+)/24: train ([\d.]+) val ([\d.]+)", seg)
        e = [int(a) for a, _, _ in ep]
        tr = [float(b) for _, b, _ in ep]; va = [float(c) for _, _, c in ep]
        ax.plot(e, tr, color=col, ls="--", lw=1, alpha=0.6)
        ax.plot(e, va, color=col, lw=1.6, marker="o", ms=3, label=f"{name} (val)")
    ax.plot([], [], color="k", ls="--", lw=1, label="train (dashed)")
    ax.set_xlabel("epoch"); ax.set_ylabel("Huber loss on log1p(PM2.5)")
    ax.set_title("Training curves — three FiLM members, split B "
                 "(early stopping: smoothed val, patience 5, min 12)")
    ax.legend(fontsize=8)
    save(fig, "fig16_training_curves.png")


def _roc(y, s):
    o = np.argsort(-s); y = np.asarray(y)[o]
    tpr = np.concatenate([[0], np.cumsum(y) / y.sum()])
    fpr = np.concatenate([[0], np.cumsum(~y) / (~y).sum()])
    return fpr, tpr, np.trapezoid(tpr, fpr)


def fig17():
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    for runs, name, col in [(SPLIT_A, "Split A", LTBLUE), (SPLIT_B, "Split B", BLUE)]:
        b = blend(runs)
        fpr, tpr, a = _roc((b.y_true > 35).values, b.y_pred.values)
        ax.plot(fpr, tpr, color=col, lw=1.6, label=f"{name} AUC={a:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.7)
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title("Exceedance detection (>35 µg/m³)\nat never-seen stations")
    ax.legend(fontsize=8.5)
    save(fig, "fig17_roc.png")


def fig18():
    bb = pd.concat([blend(SPLIT_A), blend(SPLIT_B)])
    fig, ax = plt.subplots(figsize=(6.5, 4))
    hb = ax.hexbin(bb.y_true, bb.y_pred - bb.y_true, gridsize=48, xscale="log",
                   cmap="Blues", mincnt=1)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("true PM2.5 (µg/m³, log)"); ax.set_ylabel("residual (pred − true)")
    ax.set_title("Residuals vs truth (both splits pooled) — the extreme-underprediction tail")
    fig.colorbar(hb, label="scenes")
    save(fig, "fig18_residuals.png")


def fig19():  # §3k numbers
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    arms = ["baseline", "5×", "10×"]
    ax = axes[0]
    x = np.arange(3); w = 0.36
    ax.bar(x - w/2, [-22.3, -11.7, -15.4], w, color=LTBLUE, label="35–55 bias")
    ax.bar(x + w/2, [-72.9, -49.5, -49.6], w, color=BLUE, label="55+ bias")
    ax.axhline(0, color="k", lw=0.8); ax.set_xticks(x); ax.set_xticklabels(arms)
    ax.set_ylabel("mean bias (µg/m³)"); ax.set_title("Weighted sampling: smoke bias (§3k)", fontsize=9.5)
    ax.legend(fontsize=8)
    ax = axes[1]
    overall = [0.31, 0.28, 0.26]
    ax.bar(arms, overall, color=[GRAY, LTBLUE, LTBLUE], width=0.55)
    for i, v in enumerate(overall):
        ax.text(i, v + 0.005, f"{v:.2f}", ha="center", fontweight="bold")
    ax.set_ylabel("overall spatial R²")
    ax.set_title("…at a cost to overall R² —\nthe −50 floor motivating specialists/blends", fontsize=9)
    save(fig, "fig19_weighting_sweep.png")


def fig20():
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharex=True, sharey=True)
    for ax, runs, ttl, bt in [(axes[0], SPLIT_A, "Split A", 0.503),
                              (axes[1], SPLIT_B, "Split B", 0.359)]:
        g = blend(runs).groupby("station_id")[["y_true", "y_pred"]].mean()
        ax.scatter(g.y_true, g.y_pred, s=32, color=BLUE, alpha=0.8)
        lim = [4, 20]
        ax.plot(lim, lim, "k--", lw=0.8); ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_title(f"{ttl} — between-station R²={bt:+.2f}")
        ax.set_xlabel("station true mean (µg/m³)")
    axes[0].set_ylabel("station predicted mean")
    fig.suptitle("Ranking never-seen places: predicted vs true station averages", y=1.0)
    save(fig, "fig20_between_station.png")


def fig21():
    st = pd.read_parquet("data/interim/stations.parquet").set_index("station_id")
    g = (pd.concat([blend(SPLIT_A), blend(SPLIT_B)])
         .assign(err=lambda d: (d.y_pred - d.y_true).abs()).groupby("station_id").err.mean())
    fig, ax = plt.subplots(figsize=(9, 5.2))
    sc = ax.scatter(st.loc[g.index].lon, st.loc[g.index].lat, c=g.values, s=42,
                    cmap="RdYlGn_r", vmin=1.5, vmax=8, edgecolors="k", linewidths=0.3)
    ax.set_aspect(1.25); fig.colorbar(sc, label="station MAE (µg/m³)")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title(f"Error at each held-out test station (both splits, {g.index.nunique()} stations)")
    save(fig, "fig21_station_mae_map.png")


def fig22():
    b0 = blend(SPLIT_A)
    sid = b0[b0.y_true > 55].station_id.value_counts().index[0]
    d = b0[b0.station_id == sid].sort_values("week_start")
    d = d[(d.week_start >= "2020-06-01") & (d.week_start <= "2021-01-31")]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(d.week_start, d.y_true, color="k", lw=1.2, marker="o", ms=3, label="EPA truth")
    ax.plot(d.week_start, d.y_pred, color=BLUE, lw=1.2, marker="o", ms=3, label="prediction")
    ax.axhline(35, color=RED, ls=":", lw=0.8)
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.set_title(f"2020 fire season at held-out station {sid} — event timing captured, peaks damped")
    ax.legend(fontsize=9)
    save(fig, "fig22_fireseason_timeseries.png")


def fig23():
    b1 = blend(SPLIT_B)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    ax = axes[0]
    names = ["full", "physics", "geo+time"]
    c = b1[["m0", "m1", "m2"]].corr().values
    ax.imshow(c, cmap="Blues", vmin=0.6, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{c[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="w" if c[i, j] > 0.85 else "k")
    ax.set_xticks(range(3)); ax.set_xticklabels(names, fontsize=8.5)
    ax.set_yticks(range(3)); ax.set_yticklabels(names, fontsize=8.5)
    ax.set_title("Member prediction correlation (split B)", fontsize=9.5)
    ax = axes[1]
    vals = {"Split A": ([0.377, 0.304, 0.380], 0.406), "Split B": ([0.422, 0.382, 0.299], 0.437)}
    x = np.arange(3); w = 0.35
    for off, (k, (mv, bv)), col in zip([-w/2, w/2], vals.items(), [LTBLUE, BLUE]):
        ax.bar(x + off, mv, w, color=col, label=k)
        ax.axhline(bv, color=col, ls="--", lw=1.2)
    ax.text(2.35, 0.410, "blends", fontsize=8, color="#333")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8.5)
    ax.set_ylabel("R²"); ax.set_ylim(0.25, 0.46)
    ax.set_title("Members vary by split; the blend (dashed)\nbeats the best member on both", fontsize=9)
    ax.legend(fontsize=8, loc="lower left")
    save(fig, "fig23_member_diversity.png")


def fig24():  # §3w campaign summary
    items = [("TTA (8-view averaging)", 0.017, GREEN),
             ("Temporal reference (deployable)", 0.029, GREEN),
             ("Clean reference (needs labels)", 0.059, GREEN),
             ("Chromatic band ratios", -0.006, GRAY),
             ("ResNet-34 capacity", -0.002, GRAY),
             ("Isotonic smoke calibration", -0.058, RED)]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    v = [x for _, x, _ in items]
    ax.barh(range(len(items)), v, color=[c for _, _, c in items], height=0.6)
    ax.axvline(0, color="k", lw=0.8)
    for i, x in enumerate(v):
        ax.text(x + (0.002 if x >= 0 else -0.002), i, f"{x:+.3f}",
                va="center", ha="left" if x >= 0 else "right", fontsize=8.5)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([n for n, _, _ in items], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("single-model R² change vs plain FiLM (same split)")
    ax.set_title("Improvement campaign verdicts (§3w) — what helped, what didn't")
    save(fig, "fig24_campaign_verdicts.png")


ALL = {f"fig{i}": fn for i, fn in enumerate(
    [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10, fig11, fig12, fig13,
     fig14, fig15, fig16, fig17, fig18, fig19, fig20, fig21, fig22, fig23, fig24], start=1)}

if __name__ == "__main__":
    targets = sys.argv[1:] or list(ALL)
    for t in targets:
        ALL[t]()
