"""Extract .zip archives from the discovery manifest into raw_catalog."""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

from terra_etl.ingest.manifest import (
    DiscoveryEntry,
    filter_by_extension,
    filter_zips_for_hydrography_preference,
    load_included_entries,
)
from terra_etl.ingest.models import IngestRecord, IngestReport, IngestStatus


def ingest_zips(
    manifest_path: Path | str,
    output_root: Path | str,
    *,
    entries: list[DiscoveryEntry] | None = None,
    hydrography_preferred_format: str = "fgdb",
) -> IngestReport:
    """Extract all included ``.zip`` files listed in the discovery manifest.

    Archives are extracted under ``output_root/extracted/<slug>/``. The source
    Downloads folder is never modified.

    Args:
        manifest_path: Path to ``manifest.json`` from the discover stage.
        output_root: Typically ``data/raw_catalog``.
        entries: Optional pre-filtered entries; loads from manifest when omitted.

    Returns:
        IngestReport with per-zip extraction outcomes.
    """
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    extract_root = root / "extracted"
    extract_root.mkdir(parents=True, exist_ok=True)

    loaded = entries if entries is not None else load_included_entries(manifest_path)
    loaded = filter_zips_for_hydrography_preference(loaded, hydrography_preferred_format)
    zip_entries = filter_by_extension(loaded, ".zip")

    report = IngestReport(format="zip")
    for entry in zip_entries:
        report.records.append(_extract_zip(Path(entry.path), extract_root, entry.layer_hint))

    log_path = root / "ingest_zip.json"
    log_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return report


def _extract_zip(source: Path, extract_root: Path, layer_hint: str) -> IngestRecord:
    """Extract a single zip archive with zip-slip protection."""
    if not source.is_file():
        return IngestRecord(
            source_path=str(source),
            format="zip",
            output_dir=None,
            status=IngestStatus.FAILED,
            message=f"Source file not found: {source}",
            layer_hint=layer_hint,
        )

    dest_dir = extract_root / _slugify(source.stem)
    if dest_dir.exists() and any(dest_dir.iterdir()):
        members = _list_extracted_relative(dest_dir)
        return IngestRecord(
            source_path=str(source.resolve()),
            format="zip",
            output_dir=str(dest_dir.resolve()),
            status=IngestStatus.SKIPPED,
            message="Already extracted; skipping",
            layer_hint=layer_hint,
            member_count=len(members),
            members_sample=tuple(members[:10]),
        )

    dest_dir.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(source, "r") as zf:
            members = zf.namelist()
            for member in members:
                _safe_extract_member(zf, member, dest_dir)
    except (zipfile.BadZipFile, OSError, ValueError) as exc:
        return IngestRecord(
            source_path=str(source.resolve()),
            format="zip",
            output_dir=str(dest_dir.resolve()),
            status=IngestStatus.FAILED,
            message=str(exc),
            layer_hint=layer_hint,
        )

    extracted = _list_extracted_relative(dest_dir)
    return IngestRecord(
        source_path=str(source.resolve()),
        format="zip",
        output_dir=str(dest_dir.resolve()),
        status=IngestStatus.OK,
        message=f"Extracted {len(extracted)} members",
        layer_hint=layer_hint,
        member_count=len(extracted),
        members_sample=tuple(extracted[:10]),
    )


def _safe_extract_member(zf: zipfile.ZipFile, member: str, dest_dir: Path) -> None:
    """Extract one archive member, rejecting paths that escape dest_dir."""
    target = (dest_dir / member).resolve()
    if not str(target).startswith(str(dest_dir.resolve())):
        msg = f"Zip slip detected: {member}"
        raise ValueError(msg)
    zf.extract(member, dest_dir)


def _list_extracted_relative(dest_dir: Path) -> list[str]:
    """List extracted files relative to dest_dir."""
    return sorted(
        str(p.relative_to(dest_dir)) for p in dest_dir.rglob("*") if p.is_file()
    )


def _slugify(name: str) -> str:
    """Create a stable directory name from a zip filename."""
    slug = name.lower()
    slug = re.sub(r"\s*\(\d+\)$", "", slug)
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    return slug.strip("_") or "archive"
