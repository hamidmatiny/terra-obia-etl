"""Recursively scan source_dir and filter GeoNB-relevant files."""

from __future__ import annotations

import json
import re
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

from terra_etl.config import PipelineConfig
from terra_etl.clean.csv_validate import is_province_scale_forest_csv
from terra_etl.ingest.manifest import hydrography_zip_format

# GeoNB zip internals often contain shapefile / gdb / geojson components.
_ZIP_GEOSPATIAL_SUFFIXES = (
    ".shp",
    ".dbf",
    ".shx",
    ".prj",
    ".gpkg",
    ".geojson",
    ".json",
    ".kml",
    ".kmz",
    ".gdb",
    ".csv",
    ".tif",
    ".tiff",
    ".lpk",
    ".lpkx",
)

# Loose .txt files from GeoNB export pages may be small metadata sidecars — or
# misnamed ArcGIS REST Feature Layer JSON dumps (single-line, multi-GB).
_TXT_METADATA_HINTS = ("forestry", "forest_r", "forest___")
_ESRI_JSON_PREFIX = b'{"layers":[{"layerDefinition"'
_ESRI_JSON_SNIFF_BYTES = 4096


class DiscoveryDecision(str, Enum):
    """Whether a scanned file is selected for ingestion."""

    INCLUDED = "included"
    IGNORED = "ignored"


class LayerHint(str, Enum):
    """Inferred thematic layer from filename or zip contents."""

    FOREST = "forest"
    NON_FOREST = "non_forest"
    WETLAND = "wetland"
    LIDAR = "lidar"
    HYDROGRAPHY = "hydrography"
    METADATA = "metadata"
    ESRI_JSON_EXPORT = "esri_json_export"
    PROVINCE_CSV_EXPORT = "province_csv_export"
    REDUNDANT_FORMAT_EXPORT = "redundant_format_export"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ManifestEntry:
    """Single file record in the discovery manifest."""

    path: str
    extension: str
    size_bytes: int
    decision: DiscoveryDecision
    reason: str
    layer_hint: LayerHint = LayerHint.UNKNOWN
    zip_preview: tuple[str, ...] = ()
    duplicate_of: str | None = None


@dataclass
class DiscoveryManifest:
    """Complete discovery run output."""

    source_dir: str
    scanned_at: str
    config_path: str | None
    total_scanned: int
    included: list[ManifestEntry] = field(default_factory=list)
    ignored: list[ManifestEntry] = field(default_factory=list)

    @property
    def included_count(self) -> int:
        """Number of files selected for ingestion."""
        return len(self.included)

    @property
    def ignored_count(self) -> int:
        """Number of files excluded from ingestion."""
        return len(self.ignored)

    def to_dict(self) -> dict[str, object]:
        """Serialize manifest for JSON export."""
        return {
            "source_dir": self.source_dir,
            "scanned_at": self.scanned_at,
            "config_path": self.config_path,
            "total_scanned": self.total_scanned,
            "included_count": self.included_count,
            "ignored_count": self.ignored_count,
            "included": [asdict(e) for e in self.included],
            "ignored": [asdict(e) for e in self.ignored],
        }

    def write(self, output_dir: Path) -> tuple[Path, Path]:
        """Write JSON and CSV manifests to output_dir."""
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "manifest.json"
        csv_path = output_dir / "manifest.csv"

        json_path.write_text(
            json.dumps(self.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )

        lines = [
            "decision,path,extension,size_bytes,layer_hint,reason,duplicate_of,zip_preview",
        ]
        for entry in self.included + self.ignored:
            preview = ";".join(entry.zip_preview[:10])
            lines.append(
                f"{entry.decision.value},{_csv_escape(entry.path)},{entry.extension},"
                f"{entry.size_bytes},{entry.layer_hint.value},{_csv_escape(entry.reason)},"
                f"{entry.duplicate_of or ''},{_csv_escape(preview)}"
            )
        csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return json_path, csv_path


def run_discovery(
    config: PipelineConfig,
    *,
    config_path: Path | str | None = None,
    project_root: Path | None = None,
) -> DiscoveryManifest:
    """Scan source_dir and produce a discovery manifest.

    Expected CRS/resolution assumptions:
        - Discovery is read-only; no geospatial reprojection occurs at this stage.
        - Layer hints are inferred from filenames; content sniffing is used for
          zip archives and Esri JSON ``.txt`` exports.

    Args:
        config: Pipeline configuration with source_dir and discover settings.
        config_path: Optional path to the YAML config (recorded in manifest).
        project_root: Root for resolving relative output paths.

    Returns:
        DiscoveryManifest with included and ignored file lists.
    """
    root = project_root or Path.cwd()
    resolved = config.resolve_paths(root)
    source_dir = config.source_dir.resolve()

    if not source_dir.is_dir():
        msg = f"source_dir does not exist or is not a directory: {source_dir}"
        raise FileNotFoundError(msg)

    extensions = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in config.discover.extensions}
    patterns = [_compile_pattern(p) for p in config.discover.include_patterns]
    preferred_hydro = config.hydrography.preferred_format.lower()

    candidates: list[Path] = []
    for path in source_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in extensions:
            continue
        candidates.append(path)

    entries: list[ManifestEntry] = []
    for path in sorted(candidates):
        entries.append(_classify_file(path, patterns, preferred_hydro))

    entries = _mark_duplicates(entries)
    included = [e for e in entries if e.decision == DiscoveryDecision.INCLUDED]
    ignored = [e for e in entries if e.decision == DiscoveryDecision.IGNORED]

    manifest = DiscoveryManifest(
        source_dir=str(source_dir),
        scanned_at=datetime.now(tz=UTC).isoformat(),
        config_path=str(config_path) if config_path else None,
        total_scanned=len(entries),
        included=included,
        ignored=ignored,
    )

    out_dir = Path(resolved.paths.raw_catalog)
    manifest.write(out_dir)
    return manifest


def _compile_pattern(pattern: str) -> re.Pattern[str]:
    """Compile a regex or literal substring pattern."""
    if any(c in pattern for c in r".*+?[](){}|^$\\"):
        return re.compile(pattern, re.IGNORECASE)
    return re.compile(re.escape(pattern), re.IGNORECASE)


def _name_matches(name: str, patterns: list[re.Pattern[str]]) -> bool:
    """Return True when any include pattern matches the given name."""
    return any(p.search(name) for p in patterns)


def _infer_layer(name: str) -> LayerHint:
    """Infer thematic layer from a filename or archive member path."""
    lower = name.lower()
    if re.search(r"nb_\d{4}_\d+_\d+", lower) or lower.endswith(".laz"):
        return LayerHint.LIDAR
    if "non-forest" in lower or "nonforest" in lower or "non_forest" in lower:
        return LayerHint.NON_FOREST
    if "wetland" in lower or "terres_humides" in lower:
        return LayerHint.WETLAND
    if "nbhn" in lower or "rhnb" in lower or "hydro" in lower:
        return LayerHint.HYDROGRAPHY
    if "datadictionary" in lower or "data_dictionary" in lower:
        return LayerHint.METADATA
    if "forest" in lower or "forestry" in lower or "forêt" in lower or "foret" in lower:
        return LayerHint.FOREST
    return LayerHint.UNKNOWN


def _peek_zip(path: Path, limit: int = 25) -> tuple[str, ...]:
    """List up to ``limit`` member names from a zip archive (read-only)."""
    try:
        with zipfile.ZipFile(path, "r") as zf:
            return tuple(sorted(zf.namelist())[:limit])
    except (zipfile.BadZipFile, OSError):
        return ()


def _zip_has_geospatial_payload(members: tuple[str, ...]) -> bool:
    """Return True when zip contains shapefile, GDB, or raster members."""
    for member in members:
        lower = member.lower()
        if any(lower.endswith(suffix) for suffix in _ZIP_GEOSPATIAL_SUFFIXES):
            return True
        if ".gdb/" in lower or lower.endswith(".gdb"):
            return True
    return False


def _zip_looks_geospatial(members: tuple[str, ...], patterns: list[re.Pattern[str]]) -> bool:
    """Return True when zip contents look like GeoNB geospatial data."""
    if not members or not _zip_has_geospatial_payload(members):
        return False
    for member in members:
        lower = member.lower()
        if _name_matches(lower, patterns) or _infer_layer(lower) != LayerHint.UNKNOWN:
            return True
        if ".gdb/" in lower or lower.endswith(".gdb"):
            return True
        if any(lower.endswith(s) for s in (".shp", ".gpkg", ".geojson", ".kml", ".tif")):
            return True
    return False


def _sniff_esri_json_export(path: Path) -> bool:
    """Return True when a file begins with the ArcGIS REST layers JSON prefix."""
    try:
        with path.open("rb") as fh:
            head = fh.read(_ESRI_JSON_SNIFF_BYTES)
    except OSError:
        return False
    stripped = head.lstrip()
    return stripped.startswith(_ESRI_JSON_PREFIX)


def _redundant_vector_export_reason(name: str, ext: str) -> str | None:
    """Return ignore reason when an alternate-format export duplicates shapefile zips."""
    lower = name.lower()
    layer = _infer_layer(name)

    if ext in {".geojson", ".kml"} and layer in {LayerHint.NON_FOREST, LayerHint.WETLAND}:
        return (
            f"{ext} export redundant with {layer.value} shapefile zip already ingested; "
            "same features in WGS84 web format only"
        )

    if ext == ".gpkg" and layer == LayerHint.FOREST and "forestry" in lower:
        return (
            "Regional forest GPKG redundant with shapefile zip; raw export pre-geometry-clean "
            "(interim forest GPKGs are authoritative)"
        )

    if ext == ".kmz" and layer == LayerHint.FOREST:
        return (
            "Forest KMZ is a Google Earth preview subsample (~1k features), not the full "
            "regional inventory; shapefile zip is authoritative"
        )

    return None


def _classify_file(
    path: Path,
    patterns: list[re.Pattern[str]],
    preferred_hydro_format: str = "fgdb",
) -> ManifestEntry:
    """Classify a single candidate file as included or ignored."""
    name = path.name
    ext = path.suffix.lower()
    stat = path.stat()
    layer_hint = _infer_layer(name)
    zip_preview: tuple[str, ...] = ()

    if ext == ".txt":
        if _sniff_esri_json_export(path):
            return ManifestEntry(
                path=str(path),
                extension=ext,
                size_bytes=stat.st_size,
                decision=DiscoveryDecision.IGNORED,
                reason=(
                    "ArcGIS REST Feature Layer JSON export (.txt); redundant with "
                    "shapefile/CSV/GPKG in manifest; Web Mercator (WKID 102100) vs target EPSG:32619"
                ),
                layer_hint=LayerHint.ESRI_JSON_EXPORT,
            )
        if not any(h in name.lower() for h in _TXT_METADATA_HINTS):
            return ManifestEntry(
                path=str(path),
                extension=ext,
                size_bytes=stat.st_size,
                decision=DiscoveryDecision.IGNORED,
                reason="Loose .txt file does not match GeoNB forestry metadata naming",
                layer_hint=layer_hint,
            )
        return ManifestEntry(
            path=str(path),
            extension=ext,
            size_bytes=stat.st_size,
            decision=DiscoveryDecision.INCLUDED,
            reason="GeoNB export metadata sidecar (.txt)",
            layer_hint=LayerHint.METADATA,
        )

    hydro_fmt = hydrography_zip_format(name) if ext == ".zip" else None
    if hydro_fmt and hydro_fmt != preferred_hydro_format.lower():
        return ManifestEntry(
            path=str(path),
            extension=ext,
            size_bytes=stat.st_size,
            decision=DiscoveryDecision.IGNORED,
            reason=(
                f"Hydrography {hydro_fmt} zip skipped; preferred format is "
                f"{preferred_hydro_format} (FileGDB only)"
            ),
            layer_hint=LayerHint.HYDROGRAPHY,
            zip_preview=_peek_zip(path) if ext == ".zip" else (),
        )

    if ext == ".csv" and is_province_scale_forest_csv(path):
        return ManifestEntry(
            path=str(path),
            extension=ext,
            size_bytes=stat.st_size,
            decision=DiscoveryDecision.IGNORED,
            reason=(
                "Province-scale forest CSV with embedded WKT geometry; redundant with "
                "regional shapefiles and different vintage/coverage"
            ),
            layer_hint=LayerHint.PROVINCE_CSV_EXPORT,
        )

    redundant = _redundant_vector_export_reason(name, ext)
    if redundant:
        return ManifestEntry(
            path=str(path),
            extension=ext,
            size_bytes=stat.st_size,
            decision=DiscoveryDecision.IGNORED,
            reason=redundant,
            layer_hint=LayerHint.REDUNDANT_FORMAT_EXPORT,
        )

    if _name_matches(name, patterns) or layer_hint != LayerHint.UNKNOWN:
        reason = f"Filename matches GeoNB pattern (layer_hint={layer_hint.value})"
        if ext == ".zip":
            zip_preview = _peek_zip(path)
            if zip_preview and not _zip_has_geospatial_payload(zip_preview):
                return ManifestEntry(
                    path=str(path),
                    extension=ext,
                    size_bytes=stat.st_size,
                    decision=DiscoveryDecision.IGNORED,
                    reason="Zip archive has no recognizable geospatial members",
                    layer_hint=layer_hint,
                    zip_preview=zip_preview,
                )
            reason = f"Zip matches GeoNB pattern; previewed {len(zip_preview)} members"
        return ManifestEntry(
            path=str(path),
            extension=ext,
            size_bytes=stat.st_size,
            decision=DiscoveryDecision.INCLUDED,
            reason=reason,
            layer_hint=layer_hint,
            zip_preview=zip_preview,
        )

    if ext == ".zip":
        zip_preview = _peek_zip(path)
        if _zip_looks_geospatial(zip_preview, patterns):
            layer_hint = _infer_layer(" ".join(zip_preview))
            return ManifestEntry(
                path=str(path),
                extension=ext,
                size_bytes=stat.st_size,
                decision=DiscoveryDecision.INCLUDED,
                reason="Zip contents match GeoNB geospatial patterns",
                layer_hint=layer_hint,
                zip_preview=zip_preview,
            )

    return ManifestEntry(
        path=str(path),
        extension=ext,
        size_bytes=stat.st_size,
        decision=DiscoveryDecision.IGNORED,
        reason="No GeoNB forest/non-forest/wetland filename or zip content match",
        layer_hint=layer_hint,
        zip_preview=zip_preview,
    )


def _normalize_stem(name: str) -> str:
    """Normalize filename for duplicate detection (strip copy suffixes and hash tokens)."""
    stem = Path(name).stem.lower()
    stem = re.sub(r"\s*\(\d+\)$", "", stem)
    stem = re.sub(r"[-_]?\d{10,}$", "", stem)
    stem = re.sub(r"[^a-z0-9]+", "_", stem)
    return stem.strip("_")


def _mark_duplicates(entries: list[ManifestEntry]) -> list[ManifestEntry]:
    """Flag likely duplicate downloads; keep earliest path lexicographically."""
    seen: dict[tuple[str, str], str] = {}
    updated: list[ManifestEntry] = []

    for entry in entries:
        if entry.decision != DiscoveryDecision.INCLUDED:
            updated.append(entry)
            continue

        key = (entry.extension, _normalize_stem(Path(entry.path).name))
        prior = seen.get(key)
        if prior is None:
            seen[key] = entry.path
            updated.append(entry)
            continue

        updated.append(
            ManifestEntry(
                path=entry.path,
                extension=entry.extension,
                size_bytes=entry.size_bytes,
                decision=DiscoveryDecision.IGNORED,
                reason=f"Likely duplicate of {prior}",
                layer_hint=entry.layer_hint,
                zip_preview=entry.zip_preview,
                duplicate_of=prior,
            )
        )

    return updated


def _csv_escape(value: str) -> str:
    """Escape a value for CSV output."""
    if "," in value or '"' in value or "\n" in value:
        return '"' + value.replace('"', '""') + '"'
    return value
