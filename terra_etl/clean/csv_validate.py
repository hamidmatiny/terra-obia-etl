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
    csv_rows: int
    gpkg_rows: int
    row_count_match: bool
    stdlab_csv_unique: int
    stdlab_gpkg_unique: int
    stdlab_in_csv_not_gpkg: int
    stdlab_in_gpkg_not_csv: int
    stdlab_overlap_pct: float
    passed: bool
    message: str

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


def _normalize_std(stdlab: object) -> str | None:
    """Normalize STDLAB identifiers for cross-format joins."""
    if stdlab is None or (isinstance(stdlab, float) and pd.isna(stdlab)):
        return None
    s = str(stdlab).strip().replace(",", "")
    if not s or s.lower() in {"nan", "none"}:
        return None
    return s


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
    if "20260620" in name and "forest" in name:
        return True
    return False


def match_regional_forest_csv(path: Path) -> str | None:
    """Return region id when ``path`` is a known regional forest CSV."""
    lower = path.stem.lower()
    for pattern, region_id in _REGIONAL_FOREST_CSV_MAP:
        if pattern in lower.replace("-", "_"):
            return region_id
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

        gpkg_path = interim / "forest" / f"{region_id}.gpkg"
        if not gpkg_path.is_file():
            report.records.append(
                CsvValidationRecord(
                    csv_path=str(csv_path.resolve()),
                    gpkg_path=str(gpkg_path.resolve()),
                    region_id=region_id,
                    csv_rows=0,
                    gpkg_rows=0,
                    row_count_match=False,
                    stdlab_csv_unique=0,
                    stdlab_gpkg_unique=0,
                    stdlab_in_csv_not_gpkg=0,
                    stdlab_in_gpkg_not_csv=0,
                    stdlab_overlap_pct=0.0,
                    passed=False,
                    message=f"Missing cleaned GPKG: {gpkg_path}",
                )
            )
            continue

        csv_rows = _count_csv_rows(csv_path)
        gdf = gpd.read_file(gpkg_path, columns=["STDLAB"])
        gpkg_rows = len(gdf)

        csv_std = pd.read_csv(csv_path, usecols=["STDLAB"], dtype=str)["STDLAB"].map(_normalize_std)
        gpkg_std = gdf["STDLAB"].map(_normalize_std)
        csv_set = set(csv_std.dropna())
        gpkg_set = set(gpkg_std.dropna())

        missing_in_gpkg = csv_set - gpkg_set
        missing_in_csv = gpkg_set - csv_set
        overlap_pct = len(csv_set & gpkg_set) / max(len(gpkg_set), 1) * 100

        row_count_match = csv_rows == gpkg_rows
        stdlab_ok = not missing_in_gpkg and not missing_in_csv
        passed = row_count_match and stdlab_ok

        msg_parts = []
        if not row_count_match:
            msg_parts.append(f"row count mismatch csv={csv_rows} gpkg={gpkg_rows}")
        if missing_in_gpkg:
            msg_parts.append(f"{len(missing_in_gpkg)} STDLAB in csv not in gpkg")
        if missing_in_csv:
            msg_parts.append(f"{len(missing_in_csv)} STDLAB in gpkg not in csv")
        message = "; ".join(msg_parts) if msg_parts else "row count and STDLAB sets match"

        report.records.append(
            CsvValidationRecord(
                csv_path=str(csv_path.resolve()),
                gpkg_path=str(gpkg_path.resolve()),
                region_id=region_id,
                csv_rows=csv_rows,
                gpkg_rows=gpkg_rows,
                row_count_match=row_count_match,
                stdlab_csv_unique=len(csv_set),
                stdlab_gpkg_unique=len(gpkg_set),
                stdlab_in_csv_not_gpkg=len(missing_in_gpkg),
                stdlab_in_gpkg_not_csv=len(missing_in_csv),
                stdlab_overlap_pct=round(overlap_pct, 4),
                passed=passed,
                message=message,
            )
        )

    audit_path = interim / "validate_forest_csv.json"
    audit_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return report
