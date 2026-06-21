"""Unit tests for source discovery and manifest generation."""

from __future__ import annotations

from pathlib import Path

import pytest

from terra_etl.config import PipelineConfig
from terra_etl.discover.scanner import DiscoveryDecision, run_discovery


@pytest.fixture
def sample_source_dir(tmp_path: Path) -> Path:
    """Create a minimal fake Downloads tree with relevant and irrelevant files."""
    (tmp_path / "Forest_stand.csv").write_text("id,class\n1,forest\n")
    (tmp_path / "Wetland_Terres_humides.geojson").write_text('{"type":"FeatureCollection","features":[]}')
    (tmp_path / "tax_return_2024.pdf").write_text("not scanned")
    (tmp_path / "random_notes.txt").write_text("unrelated")
    (tmp_path / "Forestry_R_1_2_gdb_meta.txt").write_text("metadata sidecar")

    import zipfile

    zip_path = tmp_path / "Forestry_R_1_2_gdb_export.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("layers/forest.shp", b"fake")

    unrelated_zip = tmp_path / "photos_backup.zip"
    with zipfile.ZipFile(unrelated_zip, "w") as zf:
        zf.writestr("readme.txt", b"vacation")

    return tmp_path


def test_discovery_includes_geonb_files_and_ignores_unrelated(
    sample_source_dir: Path,
    tmp_path: Path,
) -> None:
    """Discovery should select GeoNB-like files and reject unrelated ones."""
    config = PipelineConfig(
        source_dir=sample_source_dir,
        paths={"raw_catalog": tmp_path / "catalog"},
        discover={
            "include_patterns": ["forest", "forestry", "wetland", "terres_humides"],
            "extensions": [".csv", ".geojson", ".txt", ".zip"],
        },
    )
    manifest = run_discovery(config, project_root=tmp_path)

    included_names = {Path(e.path).name for e in manifest.included}
    assert "Forest_stand.csv" in included_names
    assert "Wetland_Terres_humides.geojson" in included_names
    assert "Forestry_R_1_2_gdb_export.zip" in included_names
    assert "Forestry_R_1_2_gdb_meta.txt" in included_names

    ignored_names = {Path(e.path).name for e in manifest.ignored}
    assert "random_notes.txt" in ignored_names
    assert "photos_backup.zip" in ignored_names

    assert (tmp_path / "catalog" / "manifest.json").exists()


def test_discovery_excludes_esri_json_txt_exports(
    sample_source_dir: Path,
    tmp_path: Path,
) -> None:
    """Misnamed .txt ArcGIS JSON exports should be excluded and reclassified."""
    esri_txt = sample_source_dir / "Forestry_R_1_2_gdb_export.txt"
    esri_txt.write_text(
        '{"layers":[{"layerDefinition":{"name":"Forestry_R_1_2","geometryType":"esriGeometryPolygon"}}]}',
        encoding="utf-8",
    )

    config = PipelineConfig(
        source_dir=sample_source_dir,
        paths={"raw_catalog": tmp_path / "catalog"},
        discover={"include_patterns": ["forestry"], "extensions": [".txt"]},
    )
    manifest = run_discovery(config, project_root=tmp_path)

    esri_entries = [e for e in manifest.ignored if Path(e.path).name == esri_txt.name]
    assert len(esri_entries) == 1
    assert esri_entries[0].layer_hint.value == "esri_json_export"
    assert "ArcGIS REST Feature Layer JSON" in esri_entries[0].reason

    meta_entries = [e for e in manifest.included if Path(e.path).name == "Forestry_R_1_2_gdb_meta.txt"]
    assert len(meta_entries) == 1
    assert meta_entries[0].layer_hint.value == "metadata"


def test_discovery_marks_duplicate_downloads(sample_source_dir: Path, tmp_path: Path) -> None:
    """Second file with same normalized stem should be flagged as duplicate."""
    dup = sample_source_dir / "Forest_stand (1).csv"
    dup.write_text("duplicate")

    config = PipelineConfig(
        source_dir=sample_source_dir,
        paths={"raw_catalog": tmp_path / "catalog"},
        discover={"include_patterns": ["forest"], "extensions": [".csv"]},
    )
    manifest = run_discovery(config, project_root=tmp_path)

    dup_entries = [e for e in manifest.ignored if Path(e.path).name == "Forest_stand.csv"]
    assert len(dup_entries) == 1
    assert dup_entries[0].decision == DiscoveryDecision.IGNORED
    assert dup_entries[0].duplicate_of is not None
