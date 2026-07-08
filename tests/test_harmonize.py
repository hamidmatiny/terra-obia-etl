"""Tests for harmonize overlap clipping and end-to-end harmonize."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
from shapely.geometry import box
from terra_etl.harmonize.harmonize import run_harmonize
from terra_etl.harmonize.overlap import apply_priority_clipping, compute_pairwise_overlaps


def _layer(name: str, lb_cat: str, cover: str, canopy: str, geom) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "source_layer": [name],
            "source_id": ["1"],
            "lb_cat": [lb_cat],
            "cover_type": [cover],
            "canopy_closure_class": [canopy],
            "l1_ds": [float("nan")],
            "l1_sc": [float("nan")],
            "l1_vs": [float("nan")],
            "l1_pstock": [float("nan")],
            "lc_code": [0.0],
            "wri_code": [0.0],
            "spvc": [0.0],
        },
        geometry=[geom],
        crs="EPSG:32619",
    )


def _empty_layer() -> gpd.GeoDataFrame:
    """Empty GeoDataFrame with the same schema/CRS as real harmonize layers."""
    return _layer("wetland", "WL", "bog", "sparse", box(0, 0, 1, 1)).iloc[0:0].copy()


def test_priority_clipping_removes_forest_under_water() -> None:
    """Raster filter drops lower-priority features overlapping higher-priority cells."""
    water = _layer("hydro_wb", "WA", "water", "open", box(0, 0, 10, 10))
    forest = _layer("forest_r1", "FO", "conifer", "dense", box(5, 5, 15, 15))
    layers = {
        "water": water,
        "wetland": _empty_layer(),
        "forest": forest,
        "non_forest": _empty_layer(),
    }
    before = compute_pairwise_overlaps(layers, phase="before", cell_size=1.0)
    assert before.pair_areas_m2["water∩forest"] > 0

    clipped = apply_priority_clipping(layers, cell_size=1.0, overlap_threshold=0.05)
    assert clipped["forest"].empty

    after = compute_pairwise_overlaps(clipped, phase="after", cell_size=1.0)
    assert after.pair_areas_m2["water∩forest"] == 0.0


def test_run_harmonize_on_fixture_interim(tmp_path: Path) -> None:
    """Harmonize fixture interim layers into labeled training outputs."""
    interim = tmp_path / "interim"
    forest_dir = interim / "forest"
    forest_dir.mkdir(parents=True)
    gdf = gpd.GeoDataFrame(
        {
            "STDLAB": ["100"],
            "L1FUNA": ["BSPR"],
            "L2FUNA": ["BSPR"],
            "L1CC": [4],
            "L2CC": [0],
            "L1DS": ["M"],
            "L1SC": [2],
            "L1VS": [1],
            "L1PSTOCK": [0],
        },
        geometry=[box(0, 0, 100, 100)],
        crs="EPSG:3857",
    )
    gdf.to_file(forest_dir / "r6_7.gpkg", driver="GPKG")

    nf_dir = interim / "non_forest"
    nf_dir.mkdir()
    gpd.GeoDataFrame(
        {"OBJECTID": [1], "PLU": ["SET"], "LC": ["VG"]},
        geometry=[box(200, 0, 300, 100)],
        crs="EPSG:2953",
    ).to_file(nf_dir / "non_forest.gpkg", driver="GPKG")

    wl_dir = interim / "wetland"
    wl_dir.mkdir()
    gpd.GeoDataFrame(
        {"OBJECTID": [1], "WC": ["BO"], "WRI": ["SA"], "SPVC": [3]},
        geometry=[box(0, 200, 100, 300)],
        crs="EPSG:2953",
    ).to_file(wl_dir / "wetland.gpkg", driver="GPKG")

    processed = tmp_path / "processed"
    combined, report = run_harmonize(
        interim,
        tmp_path / "raw_catalog",
        processed,
        target_crs_epsg=32619,
    )
    assert len(combined) == 3
    assert report.cover_type_counts["conifer"] == 1
    assert report.cover_type_counts["developed"] == 1
    assert report.cover_type_counts["bog"] == 1
    assert (processed / "labeled_stands.gpkg").is_file()
    assert (processed / "labeled_stands_audit.json").is_file()
