"""Pairwise overlap audit and raster-based priority filtering at scale."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from rasterio.transform import from_origin

logger = logging.getLogger(__name__)

LogFn = Callable[[str], None]

# Priority order: higher number wins in overlaps.
LAYER_PRIORITY: dict[str, int] = {
    "water": 4,
    "wetland": 3,
    "forest": 2,
    "non_forest": 1,
}

LAYER_ORDER: tuple[str, ...] = ("water", "wetland", "forest", "non_forest")

PAIR_LABELS: tuple[tuple[str, str], ...] = (
    ("water", "wetland"),
    ("water", "forest"),
    ("water", "non_forest"),
    ("wetland", "forest"),
    ("wetland", "non_forest"),
    ("forest", "non_forest"),
)

DEFAULT_CELL_SIZE_M = 100.0
DEFAULT_OVERLAP_THRESHOLD = 0.05
FILTER_CHUNK_SIZE = 10_000


@dataclass
class ProvincialGrid:
    """Shared 100 m (or configured) raster grid for NB layers."""

    cell_size: float
    transform: Any
    shape: tuple[int, int]
    bounds: tuple[float, float, float, float]

    @property
    def cell_area_m2(self) -> float:
        """Return the area of one raster cell in square metres."""
        return self.cell_size * self.cell_size


@dataclass
class OverlapAudit:
    """Pairwise overlap areas (m²) before and after priority filtering."""

    phase: str
    pair_areas_m2: dict[str, float] = field(default_factory=dict)
    feature_counts: dict[str, int] = field(default_factory=dict)
    method: str = "raster_100m"

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON audit output."""
        return {
            "phase": self.phase,
            "method": self.method,
            "pair_areas_m2": self.pair_areas_m2,
            "feature_counts": self.feature_counts,
        }


def _log(msg: str, log: LogFn | None = None) -> None:
    if log is not None:
        log(msg)
    else:
        print(msg, flush=True)


def _pair_key(high: str, low: str) -> str:
    return f"{high}∩{low}"


def build_provincial_grid(
    layers: dict[str, gpd.GeoDataFrame],
    *,
    cell_size: float = DEFAULT_CELL_SIZE_M,
) -> ProvincialGrid:
    """Build a shared raster grid covering all non-empty layer bounds."""
    non_empty = [gdf for gdf in layers.values() if not gdf.empty]
    if not non_empty:
        msg = "Cannot build grid: all layers empty"
        raise ValueError(msg)
    minx = min(gdf.total_bounds[0] for gdf in non_empty)
    miny = min(gdf.total_bounds[1] for gdf in non_empty)
    maxx = max(gdf.total_bounds[2] for gdf in non_empty)
    maxy = max(gdf.total_bounds[3] for gdf in non_empty)
    width = max(1, int(np.ceil((maxx - minx) / cell_size)))
    height = max(1, int(np.ceil((maxy - miny) / cell_size)))
    transform = from_origin(minx, maxy, cell_size, cell_size)
    return ProvincialGrid(
        cell_size=cell_size,
        transform=transform,
        shape=(height, width),
        bounds=(minx, miny, maxx, maxy),
    )


def _geom_to_window(
    geom: Any,
    grid: ProvincialGrid,
) -> tuple[int, int, int, int] | None:
    """Map geometry bounds to raster row/col window clipped to the provincial grid."""
    minx, miny, maxx, maxy = geom.bounds
    g_minx, g_miny, g_maxx, g_maxy = grid.bounds
    if maxx < g_minx or minx > g_maxx or maxy < g_miny or miny > g_maxy:
        return None
    col_start = max(0, int((minx - g_minx) // grid.cell_size))
    col_stop = min(grid.shape[1], int(np.ceil((maxx - g_minx) / grid.cell_size)))
    row_start = max(0, int((g_maxy - maxy) // grid.cell_size))
    row_stop = min(grid.shape[0], int(np.ceil((g_maxy - miny) / grid.cell_size)))
    if col_start >= col_stop or row_start >= row_stop:
        return None
    return row_start, row_stop, col_start, col_stop


def burn_layer(
    gdf: gpd.GeoDataFrame,
    grid: ProvincialGrid,
    *,
    layer_name: str = "layer",
    log: LogFn | None = None,
) -> np.ndarray:
    """Rasterize polygon coverage onto the provincial grid."""
    if gdf.empty:
        _log(f"    rasterize {layer_name}: 0 features → empty grid", log)
        return np.zeros(grid.shape, dtype=np.uint8)
    started = time.perf_counter()
    shapes = ((geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty)
    burned = rasterize(
        shapes,
        out_shape=grid.shape,
        transform=grid.transform,
        fill=0,
        dtype=np.uint8,
    )
    cells = int(np.count_nonzero(burned))
    elapsed = time.perf_counter() - started
    _log(
        f"    rasterize {layer_name}: {len(gdf):,} features → "
        f"{cells:,} cells ({cells * grid.cell_area_m2 / 1e6:.1f} km²) in {elapsed:.1f}s",
        log,
    )
    return cast(np.ndarray, burned)


def burn_all_layers(
    layers: dict[str, gpd.GeoDataFrame],
    grid: ProvincialGrid,
    *,
    log: LogFn | None = None,
) -> dict[str, np.ndarray]:
    """Rasterize every layer once; reused for audit and priority filtering."""
    _log("  burning layer masks onto provincial grid...", log)
    burned: dict[str, np.ndarray] = {}
    for name in LAYER_ORDER:
        burned[name] = burn_layer(layers[name], grid, layer_name=name, log=log)
    return burned


def compute_pairwise_overlaps_from_burns(
    burned: dict[str, np.ndarray],
    layers: dict[str, gpd.GeoDataFrame],
    *,
    phase: str,
    cell_size: float,
) -> OverlapAudit:
    """Compute pairwise overlap areas from pre-burned layer masks."""
    audit = OverlapAudit(phase=phase, method=f"raster_{int(cell_size)}m")
    audit.feature_counts = {name: len(gdf) for name, gdf in layers.items()}
    cell_area = cell_size * cell_size
    for high, low in PAIR_LABELS:
        overlap_cells = int(np.count_nonzero(burned[high] & burned[low]))
        audit.pair_areas_m2[_pair_key(high, low)] = float(overlap_cells * cell_area)
    return audit


def compute_pairwise_overlaps(
    layers: dict[str, gpd.GeoDataFrame],
    *,
    phase: str,
    cell_size: float = DEFAULT_CELL_SIZE_M,
    grid: ProvincialGrid | None = None,
    burned: dict[str, np.ndarray] | None = None,
    log: LogFn | None = None,
) -> OverlapAudit:
    """Compute intersection area for each priority-relevant layer pair."""
    if burned is None:
        grid = grid or build_provincial_grid(layers, cell_size=cell_size)
        burned = burn_all_layers(layers, grid, log=log)
    assert grid is not None
    return compute_pairwise_overlaps_from_burns(burned, layers, phase=phase, cell_size=cell_size)


def _overlap_fraction_with_mask(
    geom: Any,
    higher_mask: np.ndarray,
    grid: ProvincialGrid,
) -> float:
    """Fraction of a feature's rasterized area that overlaps ``higher_mask``."""
    window = _geom_to_window(geom, grid)
    if window is None:
        return 0.0
    row_start, row_stop, col_start, col_stop = window
    height = row_stop - row_start
    width = col_stop - col_start
    minx, _, _, maxy = grid.bounds
    win_transform = from_origin(
        minx + col_start * grid.cell_size,
        maxy - row_start * grid.cell_size,
        grid.cell_size,
        grid.cell_size,
    )
    feat = rasterize(
        [(geom, 1)],
        out_shape=(height, width),
        transform=win_transform,
        fill=0,
        dtype=np.uint8,
    )
    higher_slice = higher_mask[row_start:row_stop, col_start:col_stop]
    feat_cells = int(np.count_nonzero(feat))
    if feat_cells == 0:
        return 0.0
    overlap_cells = int(np.count_nonzero(feat & higher_slice))
    return overlap_cells / feat_cells


def filter_by_higher_mask(
    gdf: gpd.GeoDataFrame,
    higher_mask: np.ndarray,
    grid: ProvincialGrid,
    *,
    layer_name: str,
    overlap_threshold: float = DEFAULT_OVERLAP_THRESHOLD,
    chunk_size: int = FILTER_CHUNK_SIZE,
    log: LogFn | None = None,
) -> gpd.GeoDataFrame:
    """Drop features whose rasterized area substantially overlaps a higher-priority mask."""
    if gdf.empty or not np.any(higher_mask):
        _log(f"  filter {layer_name}: nothing to remove (empty or no higher mask)", log)
        return gdf.copy()

    started = time.perf_counter()
    keep_idx: list[Any] = []
    dropped = 0
    total = len(gdf)
    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        chunk = gdf.iloc[start:stop]
        for idx, row in chunk.iterrows():
            frac = _overlap_fraction_with_mask(row.geometry, higher_mask, grid)
            if frac <= overlap_threshold:
                keep_idx.append(idx)
            else:
                dropped += 1
        _log(
            f"  filter {layer_name}: chunk {start:,}–{stop:,} / {total:,} "
            f"({dropped:,} dropped so far)",
            log,
        )
    elapsed = time.perf_counter() - started
    _log(
        f"  filter {layer_name}: kept {len(keep_idx):,}, dropped {dropped:,} "
        f"in {elapsed:.1f}s (threshold={overlap_threshold:.0%})",
        log,
    )
    if not keep_idx:
        return gdf.iloc[0:0].copy()
    return gdf.loc[keep_idx].reset_index(drop=True)


def apply_priority_clipping(
    layers: dict[str, gpd.GeoDataFrame],
    *,
    cell_size: float = DEFAULT_CELL_SIZE_M,
    overlap_threshold: float = DEFAULT_OVERLAP_THRESHOLD,
    grid: ProvincialGrid | None = None,
    log: LogFn | None = None,
) -> dict[str, gpd.GeoDataFrame]:
    """Apply water > wetland > forest > non_forest via raster overlap filtering."""
    _log(
        f"  raster priority filter (cell={cell_size:.0f}m, threshold={overlap_threshold:.0%})",
        log,
    )
    grid = grid or build_provincial_grid(layers, cell_size=cell_size)
    _log(
        f"  provincial grid: {grid.shape[1]:,}×{grid.shape[0]:,} cells "
        f"({grid.shape[1] * grid.shape[0] / 1e6:.1f}M) at {cell_size:.0f}m",
        log,
    )

    water = layers["water"].copy()
    water_mask = burn_layer(water, grid, layer_name="water (mask)", log=log)

    wetland = layers["wetland"].copy()
    wetland = filter_by_higher_mask(
        wetland,
        water_mask,
        grid,
        layer_name="wetland",
        overlap_threshold=overlap_threshold,
        log=log,
    )
    wetland_mask = burn_layer(wetland, grid, layer_name="wetland (mask)", log=log)
    hydro_mask = ((water_mask > 0) | (wetland_mask > 0)).astype(np.uint8)

    forest = layers["forest"].copy()
    forest = filter_by_higher_mask(
        forest,
        hydro_mask,
        grid,
        layer_name="forest",
        overlap_threshold=overlap_threshold,
        log=log,
    )
    forest_mask = burn_layer(forest, grid, layer_name="forest (mask)", log=log)
    land_mask = ((hydro_mask > 0) | (forest_mask > 0)).astype(np.uint8)

    non_forest = layers["non_forest"].copy()
    non_forest = filter_by_higher_mask(
        non_forest,
        land_mask,
        grid,
        layer_name="non_forest",
        overlap_threshold=overlap_threshold,
        log=log,
    )

    return {
        "water": water,
        "wetland": wetland,
        "forest": forest,
        "non_forest": non_forest,
    }


# Backwards-compatible alias used in tests/docs.
apply_raster_priority_filter = apply_priority_clipping
