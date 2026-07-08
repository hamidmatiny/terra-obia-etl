"""Harmonize GeoNB layers into terra-OBIA labeled training datasets."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

from terra_etl.harmonize.mapping import (
    add_shape_metrics,
    encode_l1_ds,
    encode_lc_code,
    encode_numeric,
    encode_wri_code,
    map_forest_canopy,
    map_forest_cover_type,
    map_non_forest_canopy,
    map_non_forest_cover,
    map_wetland_canopy,
    map_wetland_cover,
)
from terra_etl.harmonize.overlap import (
    DEFAULT_CELL_SIZE_M,
    apply_priority_clipping,
    build_provincial_grid,
    burn_all_layers,
    compute_pairwise_overlaps_from_burns,
)

_HYDRO_LAYER = "NBHN_0000_02_Wb"
_OUTPUT_COLUMNS = [
    "source_layer",
    "source_id",
    "lb_cat",
    "cover_type",
    "canopy_closure_class",
    "area_m2",
    "perimeter_m",
    "compactness",
    "l1_ds",
    "l1_sc",
    "l1_vs",
    "l1_pstock",
    "lc_code",
    "wri_code",
    "spvc",
    "geometry",
]


@dataclass
class HarmonizeReport:
    """Audit record for harmonize stage."""

    target_crs: str
    feature_count: int
    cover_type_counts: dict[str, int] = field(default_factory=dict)
    canopy_closure_counts: dict[str, int] = field(default_factory=dict)
    overlap_before: dict[str, Any] = field(default_factory=dict)
    overlap_after: dict[str, Any] = field(default_factory=dict)
    output_gpkg: str = ""
    output_csv: str = ""
    passed: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON audit output."""
        return {
            "target_crs": self.target_crs,
            "feature_count": self.feature_count,
            "cover_type_counts": self.cover_type_counts,
            "canopy_closure_counts": self.canopy_closure_counts,
            "overlap_before": self.overlap_before,
            "overlap_after": self.overlap_after,
            "output_gpkg": self.output_gpkg,
            "output_csv": self.output_csv,
            "passed": self.passed,
        }


def find_hydro_gdb(raw_catalog: Path) -> Path | None:
    """Locate NBHN/RHNB FileGDB under ``raw_catalog/extracted/``."""
    extracted = raw_catalog / "extracted"
    if not extracted.is_dir():
        return None
    for child in sorted(extracted.iterdir()):
        if not child.is_dir():
            continue
        if "nbhn" in child.name and "rhnb" in child.name and "fgdb" in child.name:
            for gdb in child.glob("*.gdb"):
                return gdb
    return None


def _reproject(gdf: gpd.GeoDataFrame, target_crs: str) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        msg = f"Layer missing CRS: {gdf.attrs.get('source_layer', 'unknown')}"
        raise ValueError(msg)
    if str(gdf.crs) != target_crs:
        return gdf.to_crs(target_crs)
    return gdf


def _load_forest_layers(
    interim_dir: Path,
    *,
    forest_regions: list[str] | None = None,
) -> gpd.GeoDataFrame:
    forest_dir = interim_dir / "forest"
    frames: list[gpd.GeoDataFrame] = []
    for path in sorted(forest_dir.glob("*.gpkg")):
        if path.name.startswith("."):
            continue
        region = path.stem
        if forest_regions is not None and region not in forest_regions:
            continue
        gdf = gpd.read_file(path)
        gdf["source_layer"] = f"forest_{region}"
        gdf["source_id"] = gdf["STDLAB"].astype(str)
        gdf["lb_cat"] = "FO"
        gdf["cover_type"] = gdf.apply(
            lambda row: map_forest_cover_type(row.get("L1FUNA"), row.get("L2FUNA")),
            axis=1,
        )
        gdf["canopy_closure_class"] = gdf.apply(
            lambda row: map_forest_canopy(row.get("L1CC"), row.get("L2CC")),
            axis=1,
        )
        gdf["l1_ds"] = gdf["L1DS"].map(encode_l1_ds) if "L1DS" in gdf.columns else float("nan")
        gdf["l1_sc"] = gdf["L1SC"].map(encode_numeric) if "L1SC" in gdf.columns else float("nan")
        gdf["l1_vs"] = gdf["L1VS"].map(encode_numeric) if "L1VS" in gdf.columns else float("nan")
        gdf["l1_pstock"] = (
            gdf["L1PSTOCK"].map(encode_numeric) if "L1PSTOCK" in gdf.columns else float("nan")
        )
        gdf["lc_code"] = 0.0
        gdf["wri_code"] = 0.0
        gdf["spvc"] = 0.0
        frames.append(gdf)
    if not frames:
        return gpd.GeoDataFrame(
            columns=[*_OUTPUT_COLUMNS],
            geometry="geometry",
            crs="EPSG:32619",
        )
    return pd.concat(frames, ignore_index=True)


def _load_non_forest(interim_dir: Path) -> gpd.GeoDataFrame:
    path = interim_dir / "non_forest" / "non_forest.gpkg"
    if not path.is_file():
        return gpd.GeoDataFrame(columns=[*_OUTPUT_COLUMNS], geometry="geometry", crs="EPSG:32619")
    gdf = gpd.read_file(path)
    gdf["source_layer"] = "non_forest"
    gdf["source_id"] = gdf["OBJECTID"].astype(str)
    gdf["lb_cat"] = "NF"
    gdf["cover_type"] = gdf.apply(
        lambda row: map_non_forest_cover(row.get("PLU"), row.get("LC")),
        axis=1,
    )
    gdf["canopy_closure_class"] = gdf.apply(
        lambda row: map_non_forest_canopy(row.get("PLU"), row.get("LC")),
        axis=1,
    )
    gdf["l1_ds"] = float("nan")
    gdf["l1_sc"] = float("nan")
    gdf["l1_vs"] = float("nan")
    gdf["l1_pstock"] = float("nan")
    gdf["lc_code"] = gdf["LC"].map(encode_lc_code)
    gdf["wri_code"] = 0.0
    gdf["spvc"] = 0.0
    return gdf


def _load_wetland(interim_dir: Path) -> gpd.GeoDataFrame:
    path = interim_dir / "wetland" / "wetland.gpkg"
    if not path.is_file():
        return gpd.GeoDataFrame(columns=[*_OUTPUT_COLUMNS], geometry="geometry", crs="EPSG:32619")
    gdf = gpd.read_file(path)
    gdf["source_layer"] = "wetland"
    gdf["source_id"] = gdf["OBJECTID"].astype(str)
    gdf["lb_cat"] = "WL"
    gdf["cover_type"] = gdf["WC"].map(map_wetland_cover)
    gdf["canopy_closure_class"] = gdf.apply(
        lambda row: map_wetland_canopy(row.get("WC"), row.get("WRI")),
        axis=1,
    )
    gdf["l1_ds"] = float("nan")
    gdf["l1_sc"] = float("nan")
    gdf["l1_vs"] = float("nan")
    gdf["l1_pstock"] = float("nan")
    gdf["lc_code"] = 0.0
    gdf["wri_code"] = gdf["WRI"].map(encode_wri_code)
    gdf["spvc"] = gdf["SPVC"].map(encode_numeric) if "SPVC" in gdf.columns else float("nan")
    return gdf


def _load_hydro(raw_catalog: Path) -> gpd.GeoDataFrame:
    gdb = find_hydro_gdb(raw_catalog)
    if gdb is None:
        return gpd.GeoDataFrame(columns=[*_OUTPUT_COLUMNS], geometry="geometry", crs="EPSG:32619")
    gdf = gpd.read_file(gdb, layer=_HYDRO_LAYER)
    id_col = "WATERID" if "WATERID" in gdf.columns else "NID"
    gdf["source_layer"] = "hydro_wb"
    gdf["source_id"] = gdf[id_col].astype(str)
    gdf["lb_cat"] = "WA"
    gdf["cover_type"] = "water"
    gdf["canopy_closure_class"] = "open"
    gdf["l1_ds"] = float("nan")
    gdf["l1_sc"] = float("nan")
    gdf["l1_vs"] = float("nan")
    gdf["l1_pstock"] = float("nan")
    gdf["lc_code"] = 0.0
    gdf["wri_code"] = 0.0
    gdf["spvc"] = 0.0
    return gdf


def _select_output_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    for col in _OUTPUT_COLUMNS:
        if col not in gdf.columns and col != "geometry":
            gdf[col] = (
                float("nan")
                if col
                not in {
                    "source_layer",
                    "source_id",
                    "lb_cat",
                    "cover_type",
                    "canopy_closure_class",
                }
                else ""
            )
    return gdf[_OUTPUT_COLUMNS]


def _clip_layers_to_forest_bounds(
    layers: dict[str, gpd.GeoDataFrame],
) -> dict[str, gpd.GeoDataFrame]:
    """Restrict all layers to the forest union bounds (for subset dry runs)."""
    forest = layers.get("forest")
    if forest is None or forest.empty:
        return layers
    minx, miny, maxx, maxy = forest.total_bounds
    clipped: dict[str, gpd.GeoDataFrame] = {}
    for name, gdf in layers.items():
        if gdf.empty:
            clipped[name] = gdf.copy()
            continue
        subset = gdf.cx[minx:maxx, miny:maxy].copy()
        clipped[name] = subset.reset_index(drop=True)
    return clipped


def run_harmonize(
    interim_dir: Path | str,
    raw_catalog: Path | str,
    processed_dir: Path | str,
    *,
    target_crs_epsg: int = 32619,
    forest_regions: list[str] | None = None,
    clip_to_forest_bounds: bool = False,
    cell_size_m: float = DEFAULT_CELL_SIZE_M,
    output_basename: str = "labeled_stands",
) -> tuple[gpd.GeoDataFrame, HarmonizeReport]:
    """Build labeled harmonized GeoPackage and CSV for stand classifier training."""
    interim = Path(interim_dir)
    catalog = Path(raw_catalog)
    processed = Path(processed_dir)
    processed.mkdir(parents=True, exist_ok=True)
    target_crs = f"EPSG:{target_crs_epsg}"
    run_started = time.perf_counter()

    print("Loading forest layers...", flush=True)
    t0 = time.perf_counter()
    forest = _reproject(_load_forest_layers(interim, forest_regions=forest_regions), target_crs)
    print(f"  {len(forest):,} forest features ({time.perf_counter() - t0:.1f}s)", flush=True)
    if forest_regions:
        print(f"  forest subset: {', '.join(forest_regions)}", flush=True)

    print("Loading non-forest...", flush=True)
    t0 = time.perf_counter()
    non_forest = _reproject(_load_non_forest(interim), target_crs)
    elapsed = time.perf_counter() - t0
    print(f"  {len(non_forest):,} non-forest features ({elapsed:.1f}s)", flush=True)

    print("Loading wetland...", flush=True)
    t0 = time.perf_counter()
    wetland = _reproject(_load_wetland(interim), target_crs)
    print(f"  {len(wetland):,} wetland features ({time.perf_counter() - t0:.1f}s)", flush=True)

    print("Loading hydrography...", flush=True)
    t0 = time.perf_counter()
    water = _reproject(_load_hydro(catalog), target_crs)
    print(f"  {len(water):,} waterbody features ({time.perf_counter() - t0:.1f}s)", flush=True)

    layers = {
        "water": water,
        "wetland": wetland,
        "forest": forest,
        "non_forest": non_forest,
    }
    if clip_to_forest_bounds and not forest.empty:
        print("Clipping layers to forest subset bounds...", flush=True)
        layers = _clip_layers_to_forest_bounds(layers)
        for name, gdf in layers.items():
            print(f"  {name}: {len(gdf):,} features", flush=True)

    print(f"Building provincial grid ({cell_size_m:.0f} m cells)...", flush=True)
    t0 = time.perf_counter()
    grid = build_provincial_grid(layers, cell_size=cell_size_m)
    print(
        f"  grid {grid.shape[1]:,}×{grid.shape[0]:,} cells ({time.perf_counter() - t0:.1f}s)",
        flush=True,
    )

    print("Computing pre-clip overlap audit (raster)...", flush=True)
    t0 = time.perf_counter()
    burned_before = burn_all_layers(layers, grid)
    overlap_before = compute_pairwise_overlaps_from_burns(
        burned_before,
        layers,
        phase="before_clipping",
        cell_size=cell_size_m,
    )
    print(f"  pre-clip audit done ({time.perf_counter() - t0:.1f}s)", flush=True)

    print("Applying raster priority filter (water > wetland > forest > non_forest)...", flush=True)
    t0 = time.perf_counter()
    clipped = apply_priority_clipping(layers, cell_size=cell_size_m, grid=grid)
    print(f"  priority filter done ({time.perf_counter() - t0:.1f}s)", flush=True)

    print("Computing post-clip overlap audit (raster)...", flush=True)
    t0 = time.perf_counter()
    burned_after = burn_all_layers(clipped, grid)
    overlap_after = compute_pairwise_overlaps_from_burns(
        burned_after,
        clipped,
        phase="after_clipping",
        cell_size=cell_size_m,
    )
    print(f"  post-clip audit done ({time.perf_counter() - t0:.1f}s)", flush=True)

    print("Computing shape metrics and writing outputs...", flush=True)
    t0 = time.perf_counter()
    combined = pd.concat(
        [clipped["water"], clipped["wetland"], clipped["forest"], clipped["non_forest"]],
        ignore_index=True,
    )
    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs=target_crs)
    combined = add_shape_metrics(combined)
    combined = _select_output_columns(combined)

    gpkg_path = processed / f"{output_basename}.gpkg"
    csv_path = processed / f"{output_basename}.csv"
    combined.to_file(gpkg_path, driver="GPKG")
    combined.drop(columns="geometry").to_csv(csv_path, index=False)
    print(f"  wrote {len(combined):,} features ({time.perf_counter() - t0:.1f}s)", flush=True)

    report = HarmonizeReport(
        target_crs=target_crs,
        feature_count=len(combined),
        cover_type_counts=combined["cover_type"].value_counts().to_dict(),
        canopy_closure_counts=combined["canopy_closure_class"].value_counts().to_dict(),
        overlap_before=overlap_before.to_dict(),
        overlap_after=overlap_after.to_dict(),
        output_gpkg=str(gpkg_path.resolve()),
        output_csv=str(csv_path.resolve()),
    )
    audit_path = processed / f"{output_basename}_audit.json"
    audit_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    elapsed = time.perf_counter() - run_started
    print(f"Harmonize complete in {elapsed:.1f}s ({elapsed / 60:.1f} min)", flush=True)
    return combined, report
