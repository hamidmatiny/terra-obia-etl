"""Tests for regional forest CSV validation."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon

from terra_etl.clean.csv_validate import (
    is_province_scale_forest_csv,
    match_regional_forest_csv,
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
