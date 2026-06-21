"""Tests for harmonize label mapping."""

from __future__ import annotations

from terra_etl.harmonize.mapping import (
    map_forest_canopy,
    map_forest_cover_type,
    map_non_forest_canopy,
    map_non_forest_cover,
    map_wetland_canopy,
    map_wetland_cover,
)


def test_forest_funa_mapping() -> None:
    assert map_forest_cover_type("BSPR", None) == "conifer"
    assert map_forest_cover_type("IHHW", None) == "deciduous"
    assert map_forest_cover_type("BFMX", None) == "mixed"
    assert map_forest_cover_type("DFDS", None) == "conifer"
    assert map_forest_cover_type("UKWN", "INMX") == "mixed"


def test_forest_canopy_mapping() -> None:
    assert map_forest_canopy(0, None) == "open"
    assert map_forest_canopy(1, None) == "sparse"
    assert map_forest_canopy(2, None) == "moderate"
    assert map_forest_canopy(4, None) == "dense"
    assert map_forest_canopy(None, 3) == "moderate"


def test_non_forest_mapping() -> None:
    assert map_non_forest_cover("SET", "VT") == "developed"
    assert map_non_forest_cover("WIL", "NV") == "barren"
    assert map_non_forest_cover("AGR", "VG") == "agriculture"
    assert map_non_forest_canopy("SET", "VT") == "open"
    assert map_non_forest_canopy("WIL", "VT") == "moderate"


def test_wetland_mapping() -> None:
    assert map_wetland_cover("BO") == "bog"
    assert map_wetland_cover("FW") == "wetland_forest"
    assert map_wetland_canopy("FW", "SA") == "sparse"
    assert map_wetland_canopy("FW", "") == "moderate"
    assert map_wetland_canopy("AB", "PF") == "open"
