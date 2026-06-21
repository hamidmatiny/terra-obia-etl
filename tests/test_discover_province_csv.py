"""Discovery tests for province-scale forest CSV exclusion."""

from __future__ import annotations

from pathlib import Path

from terra_etl.config import PipelineConfig
from terra_etl.discover.scanner import run_discovery


def test_discovery_excludes_province_scale_forest_csv(tmp_path: Path) -> None:
    """Large province forest CSV should be excluded; regional CSV kept."""
    (tmp_path / "Forest___Forêt_20260620.csv").write_text("the_geom,STDLAB\n")
    (tmp_path / "Forest_R6_7_export.csv").write_text("STDLAB,SITEI\n")

    config = PipelineConfig(
        source_dir=tmp_path,
        paths={"raw_catalog": tmp_path / "catalog"},
        discover={"include_patterns": ["forest"], "extensions": [".csv"]},
    )
    manifest = run_discovery(config, project_root=tmp_path)

    included = {Path(e.path).name for e in manifest.included}
    ignored = {Path(e.path).name: e.reason for e in manifest.ignored}

    assert "Forest_R6_7_export.csv" in included
    assert "Forest___Forêt_20260620.csv" in ignored
    assert "Province-scale forest CSV" in ignored["Forest___Forêt_20260620.csv"]
