"""GeoNB → terra-OBIA label and inventory-attribute mapping."""

from __future__ import annotations

import math
from typing import Any

import geopandas as gpd

# FO_LxFUNA_2018 (2020 algorithm, 48 FUNAs) — Cover Type group per code.
_FUNA_GROUP: dict[str, str] = {
    "EHSW": "SW",
    "ECSW": "SW",
    "RPSW": "SW",
    "WPSW": "SW",
    "JPSW": "SW",
    "NSPR": "SW",
    "RSPR": "SW",
    "WSPR": "SW",
    "PSBS": "SW",
    "BSPR": "SW",
    "SPRC": "SW",
    "BFIR": "SW",
    "TLSW": "SW",
    "BFSP": "SW",
    "BSBF": "SW",
    "RSBF": "SW",
    "SPBF": "SW",
    "TOSW": "SW",
    "INSW": "SW",
    "DFDS": "SW",
    "DEAD": "SW",
    "OKHW": "HW",
    "FPHW": "HW",
    "BETH": "HW",
    "SMTH": "HW",
    "YBTH": "HW",
    "TOHW": "HW",
    "THIH": "HW",
    "RMHW": "HW",
    "POHW": "HW",
    "BIHW": "HW",
    "NCOM": "HW",
    "THHW": "HW",
    "IHHW": "HW",
    "OKMX": "MW",
    "EHMX": "MW",
    "ECMX": "MW",
    "PIMX": "MW",
    "THMX": "MW",
    "RSMX": "MW",
    "SPMX": "MW",
    "BFMX": "MW",
    "RMMX": "MW",
    "POMX": "MW",
    "BIMX": "MW",
    "TOMX": "MW",
    "INMX": "MW",
    "UKWN": "UK",
}

_GROUP_TO_COVER: dict[str, str] = {
    "SW": "conifer",
    "HW": "deciduous",
    "MW": "mixed",
}

# FO_CC crown closure → terra-OBIA four-class canopy bins.
_CC_TO_CANOPY: dict[int, str] = {
    0: "open",
    1: "sparse",
    2: "moderate",
    3: "moderate",
    4: "dense",
    5: "dense",
}

# NonForest PLU → cover_type (granular taxonomy).
_PLU_TO_COVER: dict[str, str] = {
    "SET": "developed",
    "AGR": "agriculture",
    "INF": "infrastructure",
    "IND": "industrial",
    "REC": "recreational",
    "DND": "defense",
}

# Wetland WC → cover_type (granular WC-derived classes).
_WC_TO_COVER: dict[str, str] = {
    "SB": "wetland_shrub",
    "FM": "wetland_marsh",
    "CM": "coastal_marsh",
    "FE": "fen",
    "BO": "bog",
    "FW": "wetland_forest",
    "AB": "aquatic_bed",
    "BC": "beach",
    "DU": "dune",
    "RK": "rocky_shore",
    "TF": "tidal_flat",
    "NP": "wetland_unknown",
    "WL": "wetland_unknown",
}

_WRI_TO_CODE: dict[str, int] = {"PF": 0, "SA": 1, "SF": 2, "TD": 3}
_LC_TO_CODE: dict[str, int] = {"NV": 0, "VG": 1, "VS": 2, "VT": 3}
_L1DS_TO_CODE: dict[str, int] = {"Y": 0, "I": 1, "M": 2, "O": 3}

_ANTHRO_PLU = frozenset({"SET", "INF", "IND", "REC", "DND"})


def _clean_str(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _cc_int(value: Any) -> int | None:
    text = _clean_str(value)
    if not text or text.lower() == "nan":
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def map_forest_cover_type(l1funa: Any, l2funa: Any) -> str:
    """Map forest FUNA codes to ``cover_type`` via FO_LxFUNA_2018."""
    code = _clean_str(l1funa)
    if code == "UKWN":
        code = _clean_str(l2funa)
    group = _FUNA_GROUP.get(code)
    if group is None or group == "UK":
        return "conifer" if code in ("DFDS", "DEAD") else "mixed"
    return _GROUP_TO_COVER[group]


def map_forest_canopy(l1cc: Any, l2cc: Any) -> str:
    """Map FO_CC crown closure codes to ``canopy_closure_class``."""
    cc = _cc_int(l1cc)
    if cc is None:
        cc = _cc_int(l2cc)
    if cc is None:
        return "moderate"
    return _CC_TO_CANOPY.get(cc, "moderate")


def map_non_forest_cover(plu: Any, lc: Any) -> str:
    """Map non-forest PLU/LC to ``cover_type``."""
    plu_code = _clean_str(plu)
    lc_code = _clean_str(lc)
    if plu_code == "WIL":
        return {
            "NV": "barren",
            "VG": "herbaceous",
            "VS": "shrub",
            "VT": "scrub",
        }.get(lc_code, "barren")
    return _PLU_TO_COVER.get(plu_code, "developed")


def map_non_forest_canopy(plu: Any, lc: Any) -> str:
    """Map non-forest PLU/LC to ``canopy_closure_class``."""
    plu_code = _clean_str(plu)
    if plu_code in _ANTHRO_PLU:
        return "open"
    if plu_code == "AGR":
        return "sparse"
    if plu_code == "WIL":
        return {
            "NV": "open",
            "VG": "sparse",
            "VS": "sparse",
            "VT": "moderate",
        }.get(_clean_str(lc), "open")
    return "open"


def map_wetland_cover(wc: Any) -> str:
    """Map wetland WC code to ``cover_type``."""
    return _WC_TO_COVER.get(_clean_str(wc), "wetland_unknown")


def map_wetland_canopy(wc: Any, wri: Any) -> str:
    """Map wetland WC/WRI to ``canopy_closure_class``."""
    wc_code = _clean_str(wc)
    wri_code = _clean_str(wri)
    if wri_code == "PF" or wc_code in {"AB", "BC", "RK", "TF"}:
        return "open"
    if wri_code == "TD":
        return "open"
    if wri_code == "SF":
        return "sparse"
    if wri_code == "SA":
        return "sparse"
    if wc_code in {"BO", "FW"}:
        return "moderate"
    if wc_code in {"FM", "FE", "SB", "CM"}:
        return "sparse"
    return "sparse"


def encode_l1_ds(value: Any) -> float:
    """Encode FO_L1DS development stage as ordinal float."""
    code = _L1DS_TO_CODE.get(_clean_str(value))
    return float("nan") if code is None else float(code)


def encode_lc_code(value: Any) -> float:
    """Encode non-forest LC as numeric code."""
    code = _LC_TO_CODE.get(_clean_str(value))
    return float("nan") if code is None else float(code)


def encode_wri_code(value: Any) -> float:
    """Encode wetland WRI as numeric code."""
    code = _WRI_TO_CODE.get(_clean_str(value))
    return float("nan") if code is None else float(code)


def encode_numeric(value: Any) -> float:
    """Parse inventory numeric fields; blank → NaN."""
    text = _clean_str(value)
    if not text or text.lower() == "nan":
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def add_shape_metrics(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add ``area_m2``, ``perimeter_m``, ``compactness`` from geometry column."""
    out = gdf.copy()
    geoms = out.geometry
    areas = geoms.area
    perimeters = geoms.length
    out["area_m2"] = areas
    out["perimeter_m"] = perimeters
    denom = perimeters**2
    out["compactness"] = (4.0 * math.pi * areas / denom).where(denom > 0, 0.0)
    return out
