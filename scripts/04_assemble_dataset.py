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
    parser.add_argument("--labels", choices=["weekly", "overpass"], default="weekly")
    args = parser.parse_args()

    cfg = load_config()
    product = args.product or cfg.patches["product"]
    labels = pd.read_parquet(cfg.path(f"labels_{args.labels}"))
    embeddings = pd.read_parquet(
        cfg.path("embeddings").with_name(f"embeddings_{product}.parquet"))
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
