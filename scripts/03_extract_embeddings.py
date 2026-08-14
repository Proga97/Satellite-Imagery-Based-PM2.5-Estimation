#!/usr/bin/env python
"""ResNet-18 embeddings for all downloaded patches of the configured product."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thesis.config import load_config
from thesis.features.embeddings import extract_embeddings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--product", choices=["l2a", "l1c"], default=None)
    args = parser.parse_args()

    cfg = load_config()
    product = args.product or cfg.patches["product"]
    root = cfg.path("patches_dir") / product / "weekly"

    patch_files = []
    for npy in sorted(root.glob("*/*.npy")):
        patch_files.append((npy.parent.name, npy.stem, npy))
    if not patch_files:
        print(f"no patches found under {root}")
        return 1
    print(f"{len(patch_files)} patches -> embeddings (gain={cfg.embeddings['rgb_gain']})")

    df = extract_embeddings(patch_files, cfg.embeddings["rgb_gain"], cfg.embeddings["batch_size"])
    out = cfg.path("embeddings").with_name(f"embeddings_{product}.parquet")
    df.to_parquet(out, index=False)
    print(f"wrote {out} ({len(df)} rows, {df.shape[1] - 2} dims)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
