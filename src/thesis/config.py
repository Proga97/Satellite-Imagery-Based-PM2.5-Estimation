"""Load configs/pipeline.yaml (+ region file) into one Config object.

All paths resolve relative to the repo root so scripts work from any cwd.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    raw: dict[str, Any]
    region: dict[str, Any]
    root: Path = field(default=REPO_ROOT)

    @property
    def labels(self) -> dict[str, Any]:
        return self.raw["labels"]

    @property
    def patches(self) -> dict[str, Any]:
        return self.raw["patches"]

    @property
    def embeddings(self) -> dict[str, Any]:
        return self.raw["embeddings"]

    @property
    def years(self) -> list[int]:
        return list(self.raw["years"])

    def path(self, key: str) -> Path:
        """Resolve a paths.* entry to an absolute Path (parent dirs created)."""
        p = self.root / self.raw["paths"][key]
        parent = p if not p.suffix else p.parent
        parent.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def ee_project(self) -> str | None:
        return os.environ.get("EE_PROJECT") or self.raw["gee"].get("project")

    def utm_epsg(self, lon: float) -> int:
        r = self.region
        return r["utm_epsg_west"] if lon < r["utm_zone_split_lon"] else r["utm_epsg_east"]


def load_config(pipeline_yaml: str | Path | None = None) -> Config:
    p = Path(pipeline_yaml) if pipeline_yaml else REPO_ROOT / "configs" / "pipeline.yaml"
    raw = yaml.safe_load(p.read_text())
    region_file = REPO_ROOT / "configs" / "region" / f"{raw['region']}.yaml"
    region = yaml.safe_load(region_file.read_text())
    return Config(raw=raw, region=region)
