"""Clean non-forest and wetland polygon layers from extracted shapefiles."""

from __future__ import annotations

import json
from pathlib import Path

from terra_etl.clean.geometry import (
    GeometryCleanReport,
    clean_geometries_split_policy,
    read_vector_without_autocorrect,
)

_LAYER_EXTRACT_HINTS: dict[str, tuple[str, ...]] = {
    "non_forest": ("non_forest",),
    "wetland": ("wetland",),
}


def find_layer_shapefile(extract_root: Path, layer: str) -> tuple[str, Path] | None:
    """Locate one polygon shapefile for ``layer`` under ``extract_root/extracted/``.

    Returns:
        ``(layer_id, shapefile_path)`` or None when not found.
    """
    hints = _LAYER_EXTRACT_HINTS.get(layer)
    if hints is None:
        return None

    extracted = extract_root / "extracted"
    if not extracted.is_dir():
        return None

    for extract_dir in sorted(extracted.iterdir()):
        if not extract_dir.is_dir():
            continue
        name = extract_dir.name
        if not any(h in name for h in hints):
            continue
        if layer == "wetland" and "non_forest" in name:
            continue
        shps = sorted(extract_dir.glob("*.shp"))
        if len(shps) != 1:
            continue
        return layer, shps[0]
    return None


def clean_layer_geometries(
    raw_catalog: Path | str,
    interim_dir: Path | str,
    layer: str,
) -> GeometryCleanReport:
    """Validate and repair one provincial layer; write audited GPKG to interim.

    Args:
        raw_catalog: ``data/raw_catalog`` root with ``extracted/`` shapefiles.
        interim_dir: ``data/interim`` root for cleaned outputs.
        layer: ``non_forest`` or ``wetland``.

    Returns:
        GeometryCleanReport with one audit record per layer.
    """
    catalog = Path(raw_catalog)
    interim = Path(interim_dir)
    found = find_layer_shapefile(catalog, layer)
    report = GeometryCleanReport()
    if found is None:
        return report

    layer_id, shp_path = found
    layer_out = interim / layer
    layer_out.mkdir(parents=True, exist_ok=True)
    output_path = layer_out / f"{layer_id}.gpkg"

    gdf = read_vector_without_autocorrect(shp_path)
    cleaned, stats = clean_geometries_split_policy(
        gdf,
        region_id=layer_id,
        source_path=str(shp_path.resolve()),
        output_path=str(output_path.resolve()),
        read_without_ogr_autocorrect=True,
        drop_sliver_features=False,
    )
    cleaned.to_file(output_path, driver="GPKG")
    report.records.append(stats)

    audit_path = interim / f"clean_{layer}_geometry.json"
    audit_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return report


def clean_vector_geometries(
    raw_catalog: Path | str,
    interim_dir: Path | str,
    *,
    layers: tuple[str, ...] = ("non_forest", "wetland"),
) -> GeometryCleanReport:
    """Clean all requested provincial vector layers."""
    combined = GeometryCleanReport()
    for layer in layers:
        report = clean_layer_geometries(raw_catalog, interim_dir, layer)
        combined.records.extend(report.records)
    return combined
