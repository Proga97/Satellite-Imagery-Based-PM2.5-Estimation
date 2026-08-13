"""Frozen ResNet-18 image embeddings (512-d) for Sentinel-2 RGB patches."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PREPROC_VERSION = "v1_gain{gain}"


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def preprocess(patch_u16: np.ndarray, rgb_gain: float) -> np.ndarray:
    """uint16 (H,W,3) raw S2 reflectance -> float32 (3,H,W) ImageNet-normalized.

    Zero pixels (masked / no observation) are filled with the patch mean of
    valid pixels before normalization.
    """
    x = patch_u16.astype(np.float32) / 10_000.0
    invalid = (patch_u16 == 0).all(axis=-1)
    if invalid.any() and not invalid.all():
        fill = x[~invalid].mean(axis=0)
        x[invalid] = fill
    x = np.clip(x * rgb_gain, 0.0, 1.0)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x.transpose(2, 0, 1)


def build_backbone() -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    model.eval()
    return model


@torch.no_grad()
def extract_embeddings(
    patch_files: list[tuple[str, str, Path]],  # (station_id, week_start, path)
    rgb_gain: float,
    batch_size: int = 256,
) -> pd.DataFrame:
    device = pick_device()
    model = build_backbone().to(device)

    rows = []
    for i in range(0, len(patch_files), batch_size):
        chunk = patch_files[i : i + batch_size]
        batch = np.stack([preprocess(np.load(p), rgb_gain) for _, _, p in chunk])
        feats = model(torch.from_numpy(batch).to(device)).cpu().numpy()
        for (sid, wk, _), f in zip(chunk, feats):
            rows.append({"station_id": sid, "week_start": wk,
                         **{f"emb_{j}": v for j, v in enumerate(f.astype(np.float32))}})
    df = pd.DataFrame(rows)
    df.attrs["preproc_version"] = PREPROC_VERSION.format(gain=rgb_gain)
    return df
