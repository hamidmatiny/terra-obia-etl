"""Unit tests for zip ingestion."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from terra_etl.ingest.manifest import DiscoveryEntry
from terra_etl.ingest.models import IngestStatus
from terra_etl.ingest.zip import ingest_zips


def _write_manifest(path: Path, included: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"included": included, "ignored": []}),
        encoding="utf-8",
    )


def test_ingest_zips_extracts_shapefile_members(tmp_path: Path) -> None:
    """Zip ingest should extract members under raw_catalog/extracted/."""
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    zip_path = source_dir / "Forest_R6_7_export.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("Foresty_R_6_7.shp", b"shp")
        zf.writestr("Foresty_R_6_7.dbf", b"dbf")

    catalog = tmp_path / "catalog"
    manifest = catalog / "manifest.json"
    _write_manifest(
        manifest,
        [
            {
                "path": str(zip_path),
                "extension": ".zip",
                "layer_hint": "forest",
                "size_bytes": zip_path.stat().st_size,
            }
        ],
    )

    report = ingest_zips(manifest, catalog)
    assert report.ok_count == 1
    record = report.records[0]
    assert record.status == IngestStatus.OK
    assert record.output_dir is not None
    assert (Path(record.output_dir) / "Foresty_R_6_7.shp").is_file()
    assert (catalog / "ingest_zip.json").is_file()


def test_ingest_zips_skips_already_extracted(tmp_path: Path) -> None:
    """Re-running ingest should skip populated extract directories."""
    zip_path = tmp_path / "forest.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("layer.shp", b"x")

    catalog = tmp_path / "catalog"
    manifest = catalog / "manifest.json"
    _write_manifest(
        manifest,
        [{"path": str(zip_path), "extension": ".zip", "layer_hint": "forest", "size_bytes": 1}],
    )

    first = ingest_zips(manifest, catalog)
    second = ingest_zips(manifest, catalog)
    assert first.records[0].status == IngestStatus.OK
    assert second.records[0].status == IngestStatus.SKIPPED


def test_ingest_zips_rejects_zip_slip(tmp_path: Path) -> None:
    """Malicious zip paths must not escape the extract directory."""
    zip_path = tmp_path / "evil.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("../outside.txt", b"bad")

    catalog = tmp_path / "catalog"
    manifest = catalog / "manifest.json"
    _write_manifest(
        manifest,
        [{"path": str(zip_path), "extension": ".zip", "layer_hint": "forest", "size_bytes": 1}],
    )

    report = ingest_zips(manifest, catalog)
    assert report.failed_count == 1
    assert report.records[0].status == IngestStatus.FAILED


def test_ingest_zips_only_processes_included_zip_entries(tmp_path: Path) -> None:
    """Non-zip manifest rows should not be passed to zip ingest."""
    zip_path = tmp_path / "a.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("a.shp", b"x")

    catalog = tmp_path / "catalog"
    manifest = catalog / "manifest.json"
    entries = [
        DiscoveryEntry(str(zip_path), ".zip", "forest", 1),
        DiscoveryEntry(str(tmp_path / "b.csv"), ".csv", "forest", 1),
    ]
    report = ingest_zips(manifest, catalog, entries=entries)
    assert len(report.records) == 1
