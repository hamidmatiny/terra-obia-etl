"""Validate regional tabular exports against cleaned vector sources."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd


@dataclass(frozen=True)
class CsvValidationRecord:
    """Outcome of validating one regional CSV against a cleaned GPKG."""

    csv_path: str
    gpkg_path: str
    region_id: str
    id_column: str
    csv_rows: int
    gpkg_rows: int
    row_count_match: bool
    id_csv_unique: int
    id_gpkg_unique: int
    id_in_csv_not_gpkg: int
    id_in_gpkg_not_csv: int
    id_overlap_pct: float
    passed: bool
    message: str
    # Backward-compatible aliases for forest validation consumers
    stdlab_csv_unique: int = 0
    stdlab_gpkg_unique: int = 0
    stdlab_in_csv_not_gpkg: int = 0
    stdlab_in_gpkg_not_csv: int = 0
    stdlab_overlap_pct: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON audit logs."""
        return asdict(self)


@dataclass
class CsvValidationReport:
    """Aggregated CSV validation run."""

    records: list[CsvValidationRecord] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Return True when all validation records passed."""
        return bool(self.records) and all(r.passed for r in self.records)

    def to_dict(self) -> dict[str, Any]:
        """Serialize report for JSON export."""
        return {
            "passed": self.passed,
            "records": [r.to_dict() for r in self.records],
        }


def _normalize_id(value: object) -> str | None:
    """Normalize identifier values for cross-format joins."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip().replace(",", "")
    if not s or s.lower() in {"nan", "none"}:
        return None
    return s


def _normalize_std(stdlab: object) -> str | None:
    """Normalize STDLAB identifiers for cross-format joins."""
    return _normalize_id(stdlab)


def _count_csv_rows(path: Path) -> int:
    """Count data rows in a CSV without loading full file into memory."""
    with path.open("rb") as fh:
        return sum(1 for _ in fh) - 1


# Regional forest CSV filename stem → interim GPKG region id
_REGIONAL_FOREST_CSV_MAP: tuple[tuple[str, str], ...] = (
    (r"forest_r6_7", "r6_7"),
    (r"forestry_r_1_2", "r1_2"),
)


def is_province_scale_forest_csv(path: Path) -> bool:
    """Return True for the large province-scale WKT forest CSV export."""
    name = path.name.lower()
    if not name.endswith(".csv"):
        return False
    if "forest" not in name and "forêt" not in name and "foret" not in name:
        return False
    # GeoNB province export naming convention observed in Downloads
    if re.search(r"forest.*for[eê]t|for[eê]t.*forest", name):
        return True
    return bool("20260620" in name and "forest" in name)


def match_regional_forest_csv(path: Path) -> str | None:
    """Return region id when ``path`` is a known regional forest CSV."""
    lower = path.stem.lower()
    for pattern, region_id in _REGIONAL_FOREST_CSV_MAP:
        if pattern in lower.replace("-", "_"):
            return region_id
    return None


def match_layer_tabular_csv(path: Path) -> tuple[str, str] | None:
    """Return ``(layer_id, id_column)`` for provincial tabular CSV exports."""
    lower = path.stem.lower()
    if "non-forest" in lower or "non_forest" in lower:
        return "non_forest", "OBJECTID"
    if "wetland" in lower or "terres_humides" in lower:
        return "wetland", "OBJECTID"
    return None


def validate_regional_forest_csvs(
    manifest_included_csv_paths: list[str],
    interim_dir: Path | str,
) -> CsvValidationReport:
    """Validate regional forest CSVs against cleaned GPKGs (no ingest).

    Args:
        manifest_included_csv_paths: Included CSV paths from discovery manifest.
        interim_dir: ``data/interim`` containing ``forest/{region}.gpkg``.

    Returns:
        CsvValidationReport written to ``interim/validate_forest_csv.json``.
    """
    interim = Path(interim_dir)
    report = CsvValidationReport()

    for csv_str in manifest_included_csv_paths:
        csv_path = Path(csv_str)
        if csv_path.suffix.lower() != ".csv":
            continue
        region_id = match_regional_forest_csv(csv_path)
        if region_id is None:
            continue

        report.records.append(
            _validate_csv_against_gpkg(
                csv_path=csv_path,
                gpkg_path=interim / "forest" / f"{region_id}.gpkg",
                region_id=region_id,
                id_column="STDLAB",
            )
        )

    audit_path = interim / "validate_forest_csv.json"
    audit_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return report


def _validate_csv_against_gpkg(
    *,
    csv_path: Path,
    gpkg_path: Path,
    region_id: str,
    id_column: str,
) -> CsvValidationRecord:
    """Validate one tabular CSV against a cleaned GPKG by identifier column."""
    if not gpkg_path.is_file():
        return CsvValidationRecord(
            csv_path=str(csv_path.resolve()),
            gpkg_path=str(gpkg_path.resolve()),
            region_id=region_id,
            id_column=id_column,
            csv_rows=0,
            gpkg_rows=0,
            row_count_match=False,
            id_csv_unique=0,
            id_gpkg_unique=0,
            id_in_csv_not_gpkg=0,
            id_in_gpkg_not_csv=0,
            id_overlap_pct=0.0,
            passed=False,
            message=f"Missing cleaned GPKG: {gpkg_path}",
        )

    csv_rows = _count_csv_rows(csv_path)
    gdf = gpd.read_file(gpkg_path, columns=[id_column])
    gpkg_rows = len(gdf)

    csv_ids = pd.read_csv(csv_path, usecols=[id_column], dtype=str)[id_column].map(_normalize_id)
    gpkg_ids = gdf[id_column].map(_normalize_id)
    csv_set = set(csv_ids.dropna())
    gpkg_set = set(gpkg_ids.dropna())

    missing_in_gpkg = csv_set - gpkg_set
    missing_in_csv = gpkg_set - csv_set
    overlap_pct = len(csv_set & gpkg_set) / max(len(gpkg_set), 1) * 100

    row_count_match = csv_rows == gpkg_rows
    id_ok = not missing_in_gpkg and not missing_in_csv
    passed = row_count_match and id_ok

    msg_parts = []
    if not row_count_match:
        msg_parts.append(f"row count mismatch csv={csv_rows} gpkg={gpkg_rows}")
    if missing_in_gpkg:
        msg_parts.append(f"{len(missing_in_gpkg)} {id_column} in csv not in gpkg")
    if missing_in_csv:
        msg_parts.append(f"{len(missing_in_csv)} {id_column} in gpkg not in csv")
    message = "; ".join(msg_parts) if msg_parts else f"row count and {id_column} sets match"

    return CsvValidationRecord(
        csv_path=str(csv_path.resolve()),
        gpkg_path=str(gpkg_path.resolve()),
        region_id=region_id,
        id_column=id_column,
        csv_rows=csv_rows,
        gpkg_rows=gpkg_rows,
        row_count_match=row_count_match,
        id_csv_unique=len(csv_set),
        id_gpkg_unique=len(gpkg_set),
        id_in_csv_not_gpkg=len(missing_in_gpkg),
        id_in_gpkg_not_csv=len(missing_in_csv),
        id_overlap_pct=round(overlap_pct, 4),
        passed=passed,
        message=message,
        stdlab_csv_unique=len(csv_set) if id_column == "STDLAB" else 0,
        stdlab_gpkg_unique=len(gpkg_set) if id_column == "STDLAB" else 0,
        stdlab_in_csv_not_gpkg=len(missing_in_gpkg) if id_column == "STDLAB" else 0,
        stdlab_in_gpkg_not_csv=len(missing_in_csv) if id_column == "STDLAB" else 0,
        stdlab_overlap_pct=round(overlap_pct, 4) if id_column == "STDLAB" else 0.0,
    )


def validate_non_forest_wetland_csvs(
    manifest_included_csv_paths: list[str],
    interim_dir: Path | str,
) -> CsvValidationReport:
    """Validate non-forest and wetland CSVs against cleaned GPKGs (no ingest).

    Args:
        manifest_included_csv_paths: Included CSV paths from discovery manifest.
        interim_dir: ``data/interim`` containing ``non_forest/`` and ``wetland/`` GPKGs.

    Returns:
        CsvValidationReport written to ``interim/validate_non_forest_wetland_csv.json``.
    """
    interim = Path(interim_dir)
    report = CsvValidationReport()

    for csv_str in manifest_included_csv_paths:
        csv_path = Path(csv_str)
        if csv_path.suffix.lower() != ".csv":
            continue
        matched = match_layer_tabular_csv(csv_path)
        if matched is None:
            continue
        layer_id, id_column = matched
        report.records.append(
            _validate_csv_against_gpkg(
                csv_path=csv_path,
                gpkg_path=interim / layer_id / f"{layer_id}.gpkg",
                region_id=layer_id,
                id_column=id_column,
            )
        )

    audit_path = interim / "validate_non_forest_wetland_csv.json"
    audit_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return report
