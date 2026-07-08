"""Explicit geometry validation and repair with audit logging."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import geopandas as gpd
import pandas as pd
from shapely.geometry import GeometryCollection, LinearRing, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.validation import explain_validity

DEFAULT_AREA_CHANGE_THRESHOLD = 0.01
MAX_AREA_CHANGE_LOG = 50

RepairMethod = Literal["dropped_sliver", "buffer0", "exterior_ring", "unchanged"]


@dataclass(frozen=True)
class AreaChangeRecord:
    """One feature whose area shifted beyond the audit threshold after repair."""

    feature_index: int
    id_field: str | None
    id_value: str | None
    area_before: float
    area_after: float
    pct_change: float
    repair_method: RepairMethod
    area_before_raw: float | None = None
    used_outer_shell_baseline: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON audit logs."""
        return asdict(self)


@dataclass(frozen=True)
class AreaChangeDistribution:
    """Histogram and summary stats for relative area change on repaired features."""

    count: int
    mean_pct: float
    median_pct: float
    p90_pct: float
    p99_pct: float
    max_pct: float
    buckets: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON audit logs."""
        return asdict(self)


@dataclass(frozen=True)
class GeometryCleanStats:
    """Audit record for geometry validation on one vector source."""

    region_id: str
    source_path: str
    output_path: str
    total_features_in: int
    total_features: int
    invalid_before: int
    dropped_sliver_count: int
    fixed_count: int
    repair_buffer0_count: int
    repair_exterior_ring_count: int
    empty_after_fix: int
    invalid_after: int
    read_without_ogr_autocorrect: bool
    invalid_reasons_sample: tuple[str, ...] = ()
    fixed_total_area_before: float = 0.0
    fixed_total_area_after: float = 0.0
    fixed_total_area_pct_change: float = 0.0
    outer_shell_baseline_count: int = 0
    area_change_threshold: float = DEFAULT_AREA_CHANGE_THRESHOLD
    area_change_count: int = 0
    area_changes_sample: tuple[AreaChangeRecord, ...] = ()
    area_change_distribution: AreaChangeDistribution | None = None
    worst_offender_comparison: tuple[dict[str, Any], ...] = ()
    passed: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON audit logs."""
        data = asdict(self)
        data["passed"] = self.passed
        data["area_changes_sample"] = [r.to_dict() for r in self.area_changes_sample]
        if self.area_change_distribution is not None:
            data["area_change_distribution"] = self.area_change_distribution.to_dict()
        return data


@dataclass
class GeometryCleanReport:
    """Aggregated geometry cleaning outcomes."""

    records: list[GeometryCleanStats] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Return True when every record passed post-fix validation."""
        return bool(self.records) and all(r.passed for r in self.records)

    def to_dict(self) -> dict[str, Any]:
        """Serialize report for JSON export."""
        return {
            "passed": self.passed,
            "record_count": len(self.records),
            "records": [r.to_dict() for r in self.records],
        }


@contextmanager
def ogr_skip_polygon_autocorrect() -> Iterator[None]:
    """Disable GDAL polygon ring auto-correction so invalidity is detected explicitly."""
    previous = os.environ.get("OGR_ORGANIZE_POLYGONS")
    os.environ["OGR_ORGANIZE_POLYGONS"] = "SKIP"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("OGR_ORGANIZE_POLYGONS", None)
        else:
            os.environ["OGR_ORGANIZE_POLYGONS"] = previous


def read_vector_without_autocorrect(path: str | os.PathLike[str]) -> gpd.GeoDataFrame:
    """Read a vector dataset without GDAL silently fixing polygon ring order."""
    with ogr_skip_polygon_autocorrect():
        return gpd.read_file(path)


def is_sliver_std(stdlab: object) -> bool:
    """Return True when STDLAB is zero/empty (degenerate inventory sliver)."""
    if stdlab is None or (isinstance(stdlab, float) and pd.isna(stdlab)):
        return True
    text = str(stdlab).strip().replace(",", "")
    if text in {"", "0", "nan", "None"}:
        return True
    try:
        return float(text) == 0.0
    except ValueError:
        return False


def relative_area_change(area_before: float, area_after: float) -> float:
    """Return absolute relative area change in [0, inf)."""
    if area_before == 0:
        return 0.0 if area_after == 0 else float("inf")
    return abs(area_after - area_before) / area_before


def exterior_ring_footprint(geom: BaseGeometry) -> BaseGeometry:
    """Rebuild polygon(s) from exterior rings, dropping nested inner-shell artifacts."""
    if geom.is_empty:
        return geom

    def solid_exterior(poly: Polygon) -> Polygon:
        return Polygon(LinearRing(poly.exterior.coords))

    def collect_exterior_solids(part: BaseGeometry) -> list[Polygon]:
        if part.is_empty:
            return []
        if isinstance(part, Polygon):
            return [solid_exterior(part)]
        if isinstance(part, MultiPolygon):
            return [solid_exterior(p) for p in part.geoms if not p.is_empty]
        if isinstance(part, GeometryCollection):
            solids: list[Polygon] = []
            for sub in part.geoms:
                solids.extend(collect_exterior_solids(sub))
            return solids
        return []

    parts = collect_exterior_solids(geom)
    if not parts:
        return geom
    if len(parts) == 1:
        return parts[0]

    # Nested-shell error pattern: discard parts whose centroid lies inside a larger shell.
    kept: list[Polygon] = []
    for i, poly in enumerate(parts):
        nested = False
        for j, other in enumerate(parts):
            if i != j and other.area > poly.area and other.contains(poly.centroid):
                nested = True
                break
        if not nested:
            kept.append(poly)

    if not kept:
        kept = parts
    if len(kept) == 1:
        result: BaseGeometry = kept[0]
    else:
        result = MultiPolygon(kept)

    if not result.is_valid:
        buffered = result.buffer(0)
        if buffered.is_valid:
            return buffered
    return result


def audit_baseline_area(original: BaseGeometry) -> tuple[float, float, bool]:
    """Return audit baseline area, raw area, and whether outer-shell baseline was used.

    Nested-shell invalid polygons inflate ``original.area`` by summing an outer stand
    shell with an inner artifact ring. For those features the exterior-ring footprint
    is the meaningful comparison baseline for post-repair area-change auditing.
    """
    raw = float(original.area)
    if original.is_valid:
        return raw, raw, False

    outer_shell = exterior_ring_footprint(original)
    shell_area = float(outer_shell.area)
    if outer_shell.is_valid and shell_area > 0 and raw > shell_area * (1.0 + 1e-6):
        return shell_area, raw, True
    return raw, raw, False


def repair_invalid_geometry(
    geom: BaseGeometry,
    *,
    method: Literal["buffer0", "exterior_ring"],
) -> BaseGeometry:
    """Repair one invalid geometry using the selected method."""
    if method == "buffer0":
        return geom.buffer(0)
    return exterior_ring_footprint(geom)


def _preferred_id_column(columns: list[str]) -> str | None:
    """Pick a stable identifier column when present."""
    for name in ("STDLAB", "OBJECTID", "ID", "NID", "WATERID"):
        if name in columns:
            return name
    return None


def _area_change_distribution(pct_values: list[float]) -> AreaChangeDistribution:
    """Build histogram buckets and percentile summary for area-change percentages."""
    if not pct_values:
        return AreaChangeDistribution(
            count=0,
            mean_pct=0.0,
            median_pct=0.0,
            p90_pct=0.0,
            p99_pct=0.0,
            max_pct=0.0,
            buckets={},
        )

    series = pd.Series(pct_values) * 100.0
    buckets = {
        "0%": int((series == 0).sum()),
        ">0-0.1%": int(((series > 0) & (series <= 0.1)).sum()),
        ">0.1-1%": int(((series > 0.1) & (series <= 1.0)).sum()),
        ">1-5%": int(((series > 1.0) & (series <= 5.0)).sum()),
        ">5-10%": int(((series > 5.0) & (series <= 10.0)).sum()),
        ">10-50%": int(((series > 10.0) & (series <= 50.0)).sum()),
        ">50%": int((series > 50.0).sum()),
    }
    return AreaChangeDistribution(
        count=len(pct_values),
        mean_pct=round(float(series.mean()), 4),
        median_pct=round(float(series.median()), 4),
        p90_pct=round(float(series.quantile(0.9)), 4),
        p99_pct=round(float(series.quantile(0.99)), 4),
        max_pct=round(float(series.max()), 4),
        buckets=buckets,
    )


def _compare_repair_methods_on_geom(geom: BaseGeometry) -> dict[str, Any]:
    """Return area-change percentages for buffer0 vs exterior-ring on one geometry."""
    baseline, raw, used_shell = audit_baseline_area(geom)
    b0 = geom.buffer(0)
    ext = exterior_ring_footprint(geom)
    return {
        "area_before_raw": raw,
        "area_baseline": baseline,
        "used_outer_shell_baseline": used_shell,
        "buffer0_area": float(b0.area),
        "exterior_ring_area": float(ext.area),
        "buffer0_pct_vs_baseline": relative_area_change(baseline, float(b0.area)) * 100,
        "exterior_ring_pct_vs_baseline": relative_area_change(baseline, float(ext.area)) * 100,
        "buffer0_pct_vs_raw": relative_area_change(raw, float(b0.area)) * 100,
        "exterior_ring_pct_vs_raw": relative_area_change(raw, float(ext.area)) * 100,
    }


def clean_geometries_split_policy(
    gdf: gpd.GeoDataFrame,
    *,
    region_id: str,
    source_path: str,
    output_path: str,
    read_without_ogr_autocorrect: bool = True,
    area_change_threshold: float = DEFAULT_AREA_CHANGE_THRESHOLD,
    id_column: str = "STDLAB",
    drop_sliver_features: bool = True,
) -> tuple[gpd.GeoDataFrame, GeometryCleanStats]:
    """Clean invalid polygons: optional sliver drop, buffer(0), exterior-ring fallback.

    Policy:
        1. When ``drop_sliver_features`` is True, drop features with ``STDLAB=0``.
        2. For remaining invalid features: apply ``buffer(0)``.
        3. If area change still exceeds ``area_change_threshold``, replace with
           exterior-ring reconstruction from the **original** invalid geometry.

    Area-change audit uses the outer-shell footprint as baseline when the raw
    invalid area was inflated by nested-shell topology (see ``audit_baseline_area``).

    Args:
        gdf: Input features (geometries may be invalid).
        region_id: Stable region identifier for audit logs.
        source_path: Original source path (for audit).
        output_path: Intended output path (for audit).
        read_without_ogr_autocorrect: Whether ``OGR_ORGANIZE_POLYGONS=SKIP`` was used on read.
        area_change_threshold: Switch to exterior-ring when buffer(0) exceeds this fraction.
        id_column: Column used for sliver detection when ``drop_sliver_features`` is True.
        drop_sliver_features: Drop degenerate ``STDLAB=0`` inventory slivers (forest only).

    Returns:
        Cleaned GeoDataFrame and audit statistics.

    Raises:
        RuntimeError: When invalid geometries remain after repair.
    """
    total_in = len(gdf)
    if gdf.empty:
        return gdf, GeometryCleanStats(
            region_id=region_id,
            source_path=source_path,
            output_path=output_path,
            total_features_in=0,
            total_features=0,
            invalid_before=0,
            dropped_sliver_count=0,
            fixed_count=0,
            repair_buffer0_count=0,
            repair_exterior_ring_count=0,
            empty_after_fix=0,
            invalid_after=0,
            read_without_ogr_autocorrect=read_without_ogr_autocorrect,
            passed=True,
        )

    working = gdf.copy()
    if drop_sliver_features and id_column in working.columns:
        sliver_mask = working[id_column].map(is_sliver_std)
    else:
        sliver_mask = pd.Series(False, index=working.index)
    dropped_sliver_count = int(sliver_mask.sum())
    if dropped_sliver_count:
        working = working.loc[~sliver_mask].copy()

    invalid_before_mask = ~working.geometry.is_valid
    invalid_before = int(invalid_before_mask.sum())

    reasons: list[str] = []
    if invalid_before:
        for idx in working.index[invalid_before_mask][:5]:
            reasons.append(explain_validity(working.loc[idx, "geometry"]))

    repair_buffer0_count = 0
    repair_exterior_ring_count = 0
    outer_shell_baseline_count = 0
    area_change_pcts: list[float] = []
    area_changes: list[AreaChangeRecord] = []
    total_area_before = 0.0
    total_area_after = 0.0
    worst_offender_rows: list[dict[str, Any]] = []

    id_col = _preferred_id_column(list(working.columns))

    if invalid_before:
        for idx in working.index[invalid_before_mask]:
            original: BaseGeometry = working.loc[idx, "geometry"]
            area_raw = float(original.area)
            baseline_area, _, used_shell_baseline = audit_baseline_area(original)
            if used_shell_baseline:
                outer_shell_baseline_count += 1

            buffered = repair_invalid_geometry(original, method="buffer0")
            buffered_pct = relative_area_change(area_raw, float(buffered.area))
            method: RepairMethod = "buffer0"
            final_geom = buffered

            if not buffered.is_valid or buffered_pct > area_change_threshold:
                exterior = repair_invalid_geometry(original, method="exterior_ring")
                exterior_pct = relative_area_change(area_raw, float(exterior.area))
                use_exterior = exterior.is_valid and (
                    not buffered.is_valid or exterior_pct < buffered_pct
                )
                if use_exterior:
                    method = "exterior_ring"
                    final_geom = exterior
                    repair_exterior_ring_count += 1
                else:
                    repair_buffer0_count += 1
            else:
                repair_buffer0_count += 1

            working.at[idx, "geometry"] = final_geom
            area_after = float(final_geom.area)
            audit_pct = relative_area_change(baseline_area, area_after)
            area_change_pcts.append(audit_pct)
            total_area_before += baseline_area
            total_area_after += area_after

            id_value = str(working.loc[idx, id_col]) if id_col else None
            if audit_pct > area_change_threshold:
                area_changes.append(
                    AreaChangeRecord(
                        feature_index=int(idx) if isinstance(idx, int) else hash(idx) % 10**9,
                        id_field=id_col,
                        id_value=id_value,
                        area_before=baseline_area,
                        area_after=area_after,
                        pct_change=audit_pct,
                        repair_method=method,
                        area_before_raw=area_raw if used_shell_baseline else None,
                        used_outer_shell_baseline=used_shell_baseline,
                    )
                )

            if id_col and id_value and (not drop_sliver_features or not is_sliver_std(id_value)):
                comparison = _compare_repair_methods_on_geom(original)
                comparison.update(
                    {
                        "feature_id": id_value,
                        "id_column": id_col,
                        "chosen_method": method,
                        "chosen_pct_vs_baseline": round(
                            relative_area_change(baseline_area, area_after) * 100, 4
                        ),
                        "chosen_pct_vs_raw": round(
                            relative_area_change(area_raw, area_after) * 100, 4
                        ),
                    }
                )
                worst_offender_rows.append(comparison)

    empty_mask = working.geometry.is_empty
    empty_after_fix = int(empty_mask.sum())
    if empty_after_fix:
        working = working.loc[~empty_mask].copy()

    invalid_after = int((~working.geometry.is_valid).sum())
    passed = invalid_after == 0

    area_changes.sort(key=lambda r: r.pct_change, reverse=True)
    worst_offender_rows.sort(
        key=lambda row: row.get("chosen_pct_vs_baseline", row.get("chosen_pct_vs_raw", 0)),
        reverse=True,
    )

    stats = GeometryCleanStats(
        region_id=region_id,
        source_path=source_path,
        output_path=output_path,
        total_features_in=total_in,
        total_features=len(working),
        invalid_before=invalid_before,
        dropped_sliver_count=dropped_sliver_count,
        fixed_count=invalid_before,
        repair_buffer0_count=repair_buffer0_count,
        repair_exterior_ring_count=repair_exterior_ring_count,
        empty_after_fix=empty_after_fix,
        invalid_after=invalid_after,
        read_without_ogr_autocorrect=read_without_ogr_autocorrect,
        invalid_reasons_sample=tuple(reasons),
        fixed_total_area_before=total_area_before,
        fixed_total_area_after=total_area_after,
        fixed_total_area_pct_change=relative_area_change(total_area_before, total_area_after),
        outer_shell_baseline_count=outer_shell_baseline_count,
        area_change_threshold=area_change_threshold,
        area_change_count=len(area_changes),
        area_changes_sample=tuple(area_changes[:MAX_AREA_CHANGE_LOG]),
        area_change_distribution=_area_change_distribution(area_change_pcts),
        worst_offender_comparison=tuple(worst_offender_rows[:10]),
        passed=passed,
    )

    if not passed:
        msg = (
            f"{region_id}: {invalid_after} invalid geometries remain after split-policy repair "
            f"(source={source_path})"
        )
        raise RuntimeError(msg)

    return working, stats


# Backward-compatible alias for tests
def validate_and_fix_geometries(
    gdf: gpd.GeoDataFrame,
    *,
    region_id: str,
    source_path: str,
    output_path: str,
    read_without_ogr_autocorrect: bool = True,
    area_change_threshold: float = DEFAULT_AREA_CHANGE_THRESHOLD,
) -> tuple[gpd.GeoDataFrame, GeometryCleanStats]:
    """Repair invalid geometries using the forest split policy."""
    return clean_geometries_split_policy(
        gdf,
        region_id=region_id,
        source_path=source_path,
        output_path=output_path,
        read_without_ogr_autocorrect=read_without_ogr_autocorrect,
        area_change_threshold=area_change_threshold,
    )
