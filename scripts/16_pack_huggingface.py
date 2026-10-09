#!/usr/bin/env python
"""Pack the dataset for the Hugging Face Hub and (optionally) upload it shard by shard.

Each shard is a parquet file with the metadata columns of data/processed/dataset.parquet
plus an `image` column holding the 224x224x3 uint16 patch as .npy bytes:

    import numpy as np, io
    arr = np.load(io.BytesIO(row["image"]))        # (224, 224, 3) uint16, reflectance x 10,000

All downloaded scenes are included; `keep` and `reason` say which ones the quality rules
removed, so users can apply their own rules. Shards are written one at a time and, with
--upload, deleted after a successful upload, so the local disk never holds the whole set.

    python scripts/16_pack_huggingface.py --out data/hf                 # pack only
    python scripts/16_pack_huggingface.py --out data/hf --upload --repo user/name
"""
import argparse
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from thesis.config import load_config  # noqa: E402

META = ["station_id", "scene_date", "pass_time_utc", "pm25", "n_hours", "keep", "reason", "lat", "lon",
        "region", "valid_fraction", "temp_c", "rh", "wind_speed", "precip_mm", "pressure_hpa",
        "elevation_m", "sun_elev_deg", "doy_sin", "doy_cos"]


def npy_bytes(path: Path) -> bytes:
    buf = io.BytesIO(); np.save(buf, np.load(path)); return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="data/hf", help="local folder for the shards")
    ap.add_argument("--rows-per-shard", type=int, default=2000)
    ap.add_argument("--upload", action="store_true", help="upload each shard to the Hub and delete it locally")
    ap.add_argument("--repo", default=None, help="Hub dataset id, e.g. user/sentinel2-pm25")
    ap.add_argument("--only-shard", type=int, default=None, help="write (and upload) a single shard, for testing")
    args = ap.parse_args()
    cfg = load_config()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # all downloaded scenes, with the quality decision, from the same joins as 12_build_dataset
    import subprocess
    subprocess.run([sys.executable, str(cfg.root / "scripts/12_build_dataset.py"), "--keep-rejected"], check=True,
                   stdout=subprocess.DEVNULL)
    d = pd.read_parquet(cfg.path("model_table").parent / "dataset.parquet")
    d["reason"] = d["reason"].fillna("").replace("KEEP", "") if "reason" in d else ""
    d["keep"] = d["reason"] == ""
    d = d.sort_values(["station_id", "scene_date"]).reset_index(drop=True)
    print(f"{len(d)} scenes ({int(d.keep.sum())} kept), {d.station_id.nunique()} stations")
    subprocess.run([sys.executable, str(cfg.root / "scripts/12_build_dataset.py")], check=True, stdout=subprocess.DEVNULL)

    api = None
    if args.upload:
        from huggingface_hub import HfApi
        api = HfApi(); api.create_repo(args.repo, repo_type="dataset", exist_ok=True)

    n_shards = (len(d) + args.rows_per_shard - 1) // args.rows_per_shard
    for i in range(n_shards):
        if args.only_shard is not None and i != args.only_shard:
            continue
        name = f"data/train-{i:05d}-of-{n_shards:05d}.parquet"
        local = out / Path(name).name
        if args.upload and api is not None and api.file_exists(args.repo, name, repo_type="dataset"):
            print(f"shard {i}: already on the Hub, skipping"); continue
        part = d.iloc[i * args.rows_per_shard:(i + 1) * args.rows_per_shard]
        images = [npy_bytes(cfg.root / p) for p in part["patch_path"]]
        tbl = pa.Table.from_pandas(part[META].reset_index(drop=True), preserve_index=False)
        tbl = tbl.append_column("image", pa.array(images, type=pa.binary()))
        pq.write_table(tbl, local, compression="zstd", row_group_size=200)
        print(f"shard {i + 1}/{n_shards}: {len(part)} rows, {local.stat().st_size / 1e6:.0f} MB")
        if args.upload:
            api.upload_file(path_or_fileobj=str(local), path_in_repo=name, repo_id=args.repo, repo_type="dataset",
                            commit_message=f"add {Path(name).name}")
            local.unlink(); print("   uploaded and removed locally")
    return 0


if __name__ == "__main__":
    sys.exit(main())
