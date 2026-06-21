"""Tests for regional forest CSV validation."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon

from terra_etl.clean.csv_validate import (
    is_province_scale_forest_csv,
    match_layer_tabular_csv,
    match_regional_forest_csv,
    validate_non_forest_wetland_csvs,
    validate_regional_forest_csvs,
)


def test_is_province_scale_forest_csv_detects_large_export() -> None:
    """Province-scale bilingual forest CSV should be detected."""
    assert is_province_scale_forest_csv(Path("Forest___Forêt_20260620.csv")) is True
    assert is_province_scale_forest_csv(Path("Forest_R6_7_6083432925512250799.csv")) is False


def test_match_regional_forest_csv_maps_regions() -> None:
    """Regional CSV filenames should map to interim GPKG region ids."""
    assert match_regional_forest_csv(Path("Forest_R6_7_6083432925512250799.csv")) == "r6_7"
    assert match_regional_forest_csv(Path("Forestry_R_1_2_gdb_export.csv")) == "r1_2"
    assert match_regional_forest_csv(Path("Wetland_Terres_humides.csv")) is None


def test_validate_regional_forest_csvs_passes_matching_std(tmp_path: Path) -> None:
    """Validation passes when row counts and STDLAB sets match."""
    csv_path = tmp_path / "Forest_R6_7_test.csv"
    pd.DataFrame({"STDLAB": ["100", "200"], "SITEI": ["A", "B"]}).to_csv(csv_path, index=False)

    gpkg_path = tmp_path / "forest" / "r6_7.gpkg"
    gpkg_path.parent.mkdir(parents=True)
    gdf = gpd.GeoDataFrame(
        {"STDLAB": ["100", "200"]},
        geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]), Polygon([(2, 2), (3, 2), (3, 3), (2, 2)])],
        crs="EPSG:3857",
    )
    gdf.to_file(gpkg_path, driver="GPKG")

    report = validate_regional_forest_csvs([str(csv_path)], tmp_path)
    assert report.passed is True
    assert report.records[0].row_count_match is True
    assert report.records[0].stdlab_in_csv_not_gpkg == 0


def test_match_layer_tabular_csv() -> None:
    """Non-forest and wetland CSV filenames should map to layer ids."""
    assert match_layer_tabular_csv(Path("Non-Forest_Non_forestières.csv")) == ("non_forest", "OBJECTID")
    assert match_layer_tabular_csv(Path("Wetland_Terres_humides.csv")) == ("wetland", "OBJECTID")
    assert match_layer_tabular_csv(Path("Forest_R6_7_test.csv")) is None


def test_validate_non_forest_wetland_csvs(tmp_path: Path) -> None:
    """Validation passes when row counts and OBJECTID sets match."""
    csv_path = tmp_path / "Wetland_Terres_humides.csv"
    pd.DataFrame({"OBJECTID": ["1", "2"], "WC": ["SB", "BM"]}).to_csv(csv_path, index=False)

    gpkg_path = tmp_path / "wetland" / "wetland.gpkg"
    gpkg_path.parent.mkdir(parents=True)
    gdf = gpd.GeoDataFrame(
        {"OBJECTID": [1, 2], "WC": ["SB", "BM"]},
        geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]), Polygon([(2, 2), (3, 2), (3, 3), (2, 2)])],
        crs="EPSG:2953",
    )
    gdf.to_file(gpkg_path, driver="GPKG")

    report = validate_non_forest_wetland_csvs([str(csv_path)], tmp_path)
    assert report.passed is True
    assert report.records[0].row_count_match is True
    assert report.records[0].id_in_csv_not_gpkg == 0
