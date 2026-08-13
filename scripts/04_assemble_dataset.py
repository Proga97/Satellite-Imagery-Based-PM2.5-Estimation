#!/usr/bin/env python
"""Join labels + embeddings + station metadata -> model_table.parquet."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from thesis.config import load_config
from thesis.dataset.assemble import assemble


def main() -> int:
    cfg = load_config()
    labels = pd.read_parquet(cfg.path("labels_weekly"))
    embeddings = pd.read_parquet(cfg.path("embeddings"))
    stations = pd.read_parquet(cfg.path("stations"))

    table, report = assemble(labels, embeddings, stations)
    out = cfg.path("model_table")
    table.to_parquet(out, index=False)
    out.with_suffix(".report.json").write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
