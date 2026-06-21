"""Load discovery manifest for downstream ingest stages."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DiscoveryEntry:
    """Minimal discovery manifest row for ingest."""

    path: str
    extension: str
    layer_hint: str
    size_bytes: int


def hydrography_zip_format(name: str) -> str | None:
    """Return fgdb/shp/lpk when ``name`` is an NBHN/RHNB hydrography zip, else None."""
    lower = name.lower().replace("-", "_")
    if "nbhn" not in lower and "rhnb" not in lower:
        return None
    if "fgdb" in lower:
        return "fgdb"
    if "lpk" in lower:
        return "lpk"
    if "shp" in lower:
        return "shp"
    return "unknown"


def load_included_entries(manifest_path: Path | str) -> list[DiscoveryEntry]:
    """Load included files from a discovery manifest JSON file."""
    data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return [
        DiscoveryEntry(
            path=row["path"],
            extension=row["extension"],
            layer_hint=row.get("layer_hint", "unknown"),
            size_bytes=int(row["size_bytes"]),
        )
        for row in data.get("included", [])
    ]


def filter_by_extension(entries: list[DiscoveryEntry], extension: str) -> list[DiscoveryEntry]:
    """Return manifest entries matching a file extension (e.g. ``.zip``)."""
    ext = extension if extension.startswith(".") else f".{extension}"
    return [e for e in entries if e.extension.lower() == ext.lower()]


def filter_zips_for_hydrography_preference(
    entries: list[DiscoveryEntry],
    preferred_format: str,
) -> list[DiscoveryEntry]:
    """Drop non-preferred NBHN/RHNB hydrography zip entries before ingest."""
    preferred = preferred_format.lower()
    kept: list[DiscoveryEntry] = []
    for entry in entries:
        if entry.extension.lower() != ".zip":
            kept.append(entry)
            continue
        fmt = hydrography_zip_format(Path(entry.path).name)
        if fmt and fmt != preferred:
            continue
        kept.append(entry)
    return kept
