"""YAML-backed pipeline configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class PathsConfig(BaseModel):
    """Project working directories (never the read-only source_dir)."""

    raw_catalog: Path = Path("data/raw_catalog")
    interim: Path = Path("data/interim")
    processed: Path = Path("data/processed")


class DiscoverConfig(BaseModel):
    """Discovery scan and filter settings."""

    extensions: list[str] = Field(
        default_factory=lambda: [
            ".laz",
            ".csv",
            ".xlsx",
            ".zip",
            ".kml",
            ".kmz",
            ".geojson",
            ".gpkg",
            ".txt",
        ]
    )
    include_patterns: list[str] = Field(default_factory=list)
    require_confirmation: bool = True


class HydrographyConfig(BaseModel):
    """NBHN/RHNB hydrography source format preference."""

    preferred_format: str = "fgdb"


class PipelineConfig(BaseModel):
    """Root configuration loaded from YAML."""

    source_dir: Path
    target_crs_epsg: int = 32619
    layers: list[str] = Field(default_factory=lambda: ["forest", "non_forest", "wetland"])
    paths: PathsConfig = Field(default_factory=PathsConfig)
    discover: DiscoverConfig = Field(default_factory=DiscoverConfig)
    hydrography: HydrographyConfig = Field(default_factory=HydrographyConfig)

    @classmethod
    def from_yaml(cls, path: Path | str) -> PipelineConfig:
        """Load and validate configuration from a YAML file."""
        config_path = Path(path)
        raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        return cls.model_validate(raw)

    def resolve_paths(self, project_root: Path | None = None) -> PipelineConfig:
        """Return a copy with relative paths resolved against project_root."""
        root = project_root or Path.cwd()
        data = self.model_dump()
        for key in ("raw_catalog", "interim", "processed"):
            rel = Path(data["paths"][key])
            if not rel.is_absolute():
                data["paths"][key] = str((root / rel).resolve())
        return PipelineConfig.model_validate(data)
