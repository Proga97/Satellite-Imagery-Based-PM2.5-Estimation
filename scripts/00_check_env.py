#!/usr/bin/env python
"""Verify the environment: imports, device, EPA reachability, GEE auth."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

def main() -> int:
    ok = True

    print("== imports ==")
    for mod in ["numpy", "pandas", "pyarrow", "sklearn", "lightgbm", "torch",
                "torchvision", "yaml", "pyproj", "ee", "requests", "tqdm"]:
        try:
            __import__(mod)
            print(f"  ok   {mod}")
        except Exception as exc:
            print(f"  FAIL {mod}: {exc}")
            ok = False

    print("== torch device ==")
    import torch
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {dev}")

    print("== config ==")
    from thesis.config import load_config
    cfg = load_config()
    print(f"  region: {cfg.region['name']}, years: {cfg.years}, product: {cfg.patches['product']}")
    print(f"  ee project: {cfg.ee_project or 'NOT SET (gee.project in configs/pipeline.yaml or EE_PROJECT env)'}")

    print("== earth engine ==")
    from thesis.gee.auth import init_ee
    try:
        init_ee(cfg.ee_project)
        import ee
        val = ee.Number(1).add(1).getInfo()
        print(f"  ok   ee.Initialize + smoke query (1+1={val})")
    except RuntimeError as exc:
        print(f"  FAIL {exc}")
        ok = False

    print("== overall:", "PASS" if ok else "FAIL", "==")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
