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
    region: dict[str, Any]          # first region (backward compat)
    regions: list[dict[str, Any]] = field(default_factory=list)
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
        """Standard northern-hemisphere UTM zone from longitude (region-agnostic)."""
        zone = int((lon + 180) // 6) + 1
        return 32600 + zone


def load_config(pipeline_yaml: str | Path | None = None) -> Config:
    p = Path(pipeline_yaml) if pipeline_yaml else REPO_ROOT / "configs" / "pipeline.yaml"
    raw = yaml.safe_load(p.read_text())
    names = raw.get("regions") or [raw["region"]]
    regions = [
        yaml.safe_load((REPO_ROOT / "configs" / "region" / f"{n}.yaml").read_text())
        for n in names
    ]
    return Config(raw=raw, region=regions[0], regions=regions)
