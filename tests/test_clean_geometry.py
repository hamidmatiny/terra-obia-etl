"""Unit tests for auditable geometry cleaning."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from terra_etl.clean.geometry import (
    audit_baseline_area,
    clean_geometries_split_policy,
    exterior_ring_footprint,
    is_sliver_std,
    ogr_skip_polygon_autocorrect,
    read_vector_without_autocorrect,
    validate_and_fix_geometries,
)


def test_is_sliver_std() -> None:
    """STDLAB zero/empty values should be treated as slivers."""
    assert is_sliver_std(0) is True
    assert is_sliver_std("0") is True
    assert is_sliver_std("") is True
    assert is_sliver_std(None) is True
    assert is_sliver_std("12345") is False
    assert is_sliver_std(42) is False


def test_validate_and_fix_repairs_bowtie_and_leaves_zero_invalid(tmp_path: Path) -> None:
    """Invalid bowtie polygon should be fixed with logged counts and pass validation."""
    bowtie = Polygon([(0, 0), (1, 1), (1, 0), (0, 1), (0, 0)])
    gdf = gpd.GeoDataFrame({"STDLAB": ["999"]}, geometry=[bowtie], crs="EPSG:32619")
    shp = tmp_path / "invalid.shp"
    gdf.to_file(shp)

    source = read_vector_without_autocorrect(shp)
    cleaned, stats = validate_and_fix_geometries(
        source,
        region_id="test",
        source_path=str(shp),
        output_path=str(tmp_path / "out.gpkg"),
    )

    assert stats.invalid_before >= 1
    assert stats.fixed_count >= 1
    assert stats.invalid_after == 0
    assert stats.passed is True
    assert bool(cleaned.geometry.is_valid.all())
    assert stats.area_change_distribution is not None


def test_split_policy_drops_sliver_std(tmp_path: Path) -> None:
    """STDLAB=0 features should be dropped before repair."""
    valid = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    bowtie = Polygon([(0, 0), (1, 1), (1, 0), (0, 1), (0, 0)])
    gdf = gpd.GeoDataFrame(
        {"STDLAB": ["0", "123"]},
        geometry=[valid, bowtie],
        crs="EPSG:32619",
    )
    cleaned, stats = clean_geometries_split_policy(
        gdf,
        region_id="test",
        source_path="memory",
        output_path=str(tmp_path / "out.gpkg"),
    )
    assert stats.total_features_in == 2
    assert stats.dropped_sliver_count == 1
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["STDLAB"] == "123"


def test_exterior_ring_footprint_discards_inner_ring() -> None:
    """Exterior-ring reconstruction should fill the outer boundary only."""
    outer = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
    inner = [(2, 2), (8, 2), (8, 8), (2, 8), (2, 2)]
    poly = Polygon(outer, [inner])
    rebuilt = exterior_ring_footprint(poly)
    assert rebuilt.is_valid
    assert rebuilt.area == Polygon(outer).area
    assert len(rebuilt.interiors) == 0


def test_audit_baseline_uses_outer_shell_for_nested_inflation() -> None:
    """Nested-shell invalid areas should audit against outer-shell footprint, not raw sum."""
    from shapely import from_wkt

    fixture = Path(__file__).parent / "fixtures" / "nested_shell_49523619.wkt"
    if not fixture.is_file():
        return

    geom = from_wkt(fixture.read_text())
    baseline, raw, used_shell = audit_baseline_area(geom)
    assert used_shell is True
    assert raw > baseline
    assert baseline == pytest.approx(745371.63, rel=1e-4)


def test_split_policy_repairs_real_nested_shell_fixture() -> None:
    """Real nested-shell invalid MultiPolygon should repair to the outer shell footprint."""
    from shapely import from_wkt

    fixture = Path(__file__).parent / "fixtures" / "nested_shell_49523619.wkt"
    if not fixture.is_file():
        return  # fixture omitted in minimal checkouts

    invalid = from_wkt(fixture.read_text())
    assert not invalid.is_valid

    gdf = gpd.GeoDataFrame({"STDLAB": ["49523619"]}, geometry=[invalid], crs="EPSG:3857")
    cleaned, stats = clean_geometries_split_policy(
        gdf,
        region_id="test",
        source_path="memory",
        output_path="memory",
        area_change_threshold=0.01,
    )
    assert stats.repair_exterior_ring_count == 0
    assert stats.invalid_after == 0
    assert stats.area_change_count == 0
    assert stats.outer_shell_baseline_count == 1
    assert cleaned.geometry.iloc[0].area == pytest.approx(745371.63, rel=1e-4)


def test_validate_and_fix_logs_area_change_distribution(tmp_path: Path) -> None:
    """Repaired features should populate area-change distribution stats."""
    bowtie = Polygon([(0, 0), (2, 2), (2, 0), (0, 2), (0, 0)])
    gdf = gpd.GeoDataFrame({"STDLAB": ["A"]}, geometry=[bowtie], crs="EPSG:32619")
    _, stats = validate_and_fix_geometries(
        gdf,
        region_id="test",
        source_path="memory",
        output_path=str(tmp_path / "out.gpkg"),
        area_change_threshold=0.0,
    )
    assert stats.fixed_total_area_before >= 0
    assert stats.fixed_total_area_after >= 0
    assert stats.invalid_before == 1
    assert stats.area_change_distribution is not None
    assert stats.area_change_distribution.count == 1


def test_ogr_skip_polygon_autocorrect_sets_env() -> None:
    """Context manager should set and restore OGR_ORGANIZE_POLYGONS."""
    import os

    with ogr_skip_polygon_autocorrect():
        assert os.environ.get("OGR_ORGANIZE_POLYGONS") == "SKIP"
