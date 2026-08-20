"""Join weekly labels with embeddings + station metadata into the model table.

lat/lon/week are carried as METADATA (split keys / future features), not used
as model features in phase 1.
"""
from __future__ import annotations

import pandas as pd


def assemble(labels: pd.DataFrame, embeddings: pd.DataFrame, stations: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    lab = labels.copy()
    lab["week_start"] = pd.to_datetime(lab["week_start"]).dt.strftime("%Y-%m-%d")
    emb = embeddings.copy()
    emb["week_start"] = pd.to_datetime(emb["week_start"]).dt.strftime("%Y-%m-%d")

    table = lab.merge(emb, on=["station_id", "week_start"], how="inner")
    station_cols = ["station_id", "lat", "lon"] + (["region"] if "region" in stations else [])
    table = table.merge(stations[station_cols], on="station_id", how="left")

    wk = pd.to_datetime(table["week_start"])
    table["year"] = wk.dt.year
    table["week_of_year"] = wk.dt.isocalendar().week.astype(int)

    report = {
        "n_labels": len(lab),
        "n_embeddings": len(emb),
        "n_joined": len(table),
        "labels_without_patch": len(lab) - len(table),
        "n_stations": table["station_id"].nunique(),
    }
    return table, report


def feature_columns(table: pd.DataFrame, feature_set: str) -> list[str]:
    emb_cols = [c for c in table.columns if c.startswith("emb_")]
    if feature_set == "image":
        return emb_cols
    if feature_set == "spacetime":
        return ["lat", "lon", "week_of_year"]
    if feature_set == "fused":
        return emb_cols + ["lat", "lon", "week_of_year"]
    raise ValueError(f"unknown feature set: {feature_set}")
