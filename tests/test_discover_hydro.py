"""Discovery tests for hydrography format preference."""

from __future__ import annotations

from pathlib import Path

import pytest

from terra_etl.config import PipelineConfig
from terra_etl.discover.scanner import DiscoveryDecision, run_discovery


@pytest.fixture
def hydro_source_dir(tmp_path: Path) -> Path:
    """NBHN/RHNB hydrography zips in three formats."""
    for name in (
        "geonb_nbhn-rhnb_fgdb.zip",
        "geonb_nbhn-rhnb_shp.zip",
        "geonb_nbhn-rhnb_lpk.zip",
    ):
        (tmp_path / name).write_bytes(b"placeholder")
    return tmp_path


def test_discovery_keeps_only_fgdb_hydrography_zip(hydro_source_dir: Path, tmp_path: Path) -> None:
    """Only FileGDB hydro zip should be included when preferred_format is fgdb."""
    config = PipelineConfig(
        source_dir=hydro_source_dir,
        paths={"raw_catalog": tmp_path / "catalog"},
        discover={"include_patterns": ["geonb", "nbhn", "rhnb"], "extensions": [".zip"]},
        hydrography={"preferred_format": "fgdb"},
    )
    manifest = run_discovery(config, project_root=tmp_path)

    included = {Path(e.path).name for e in manifest.included}
    ignored = {Path(e.path).name: e.reason for e in manifest.ignored}

    assert "geonb_nbhn-rhnb_fgdb.zip" in included
    assert "geonb_nbhn-rhnb_shp.zip" in ignored
    assert "geonb_nbhn-rhnb_lpk.zip" in ignored
    assert "FileGDB only" in ignored["geonb_nbhn-rhnb_shp.zip"]
    assert ignored["geonb_nbhn-rhnb_shp.zip"] and manifest.ignored[0].layer_hint.value == "hydrography"
