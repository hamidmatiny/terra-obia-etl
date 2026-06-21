"""Clean forest inventory polygons from extracted shapefiles."""

from __future__ import annotations

import json
import re
from pathlib import Path

from terra_etl.clean.geometry import (
    GeometryCleanReport,
    clean_geometries_split_policy,
    read_vector_without_autocorrect,
)
from terra_etl.ingest.zip import _slugify


def _region_id_from_extract_dir(extract_dir: Path, shapefile: Path) -> str:
    """Derive a short region id from shapefile stem numeric tokens (e.g. R_3_4_5 → r3_4_5)."""
    nums = re.findall(r"\d+", shapefile.stem)
    if nums:
        return "r" + "_".join(nums)
    return extract_dir.name


def find_forest_shapefiles(extract_root: Path) -> list[tuple[str, Path, Path]]:
    """Locate forest polygon shapefiles under ``extract_root/extracted/``.

    Returns:
        List of ``(region_id, extract_dir, shapefile_path)`` tuples.
    """
    results: list[tuple[str, Path, Path]] = []
    extracted = extract_root / "extracted"
    if not extracted.is_dir():
        return results

    for extract_dir in sorted(extracted.iterdir()):
        if not extract_dir.is_dir():
            continue
        name = extract_dir.name
        if "forest" not in name or "non_forest" in name:
            continue
        shps = sorted(extract_dir.glob("*.shp"))
        if len(shps) != 1:
            continue
        shp = shps[0]
        region_id = _region_id_from_extract_dir(extract_dir, shp)
        results.append((region_id, extract_dir, shp))
    return results


def clean_forest_geometries(
    raw_catalog: Path | str,
    interim_dir: Path | str,
) -> GeometryCleanReport:
    """Validate and repair forest polygons; write audited GPKG outputs to interim.

    Expected CRS/resolution assumptions:
        - Inputs are forest inventory shapefiles in their native CRS (EPSG:3857 today).
        - Reprojection to EPSG:32619 happens in a later harmonize stage.

    Args:
        raw_catalog: ``data/raw_catalog`` root with ``extracted/`` shapefiles.
        interim_dir: ``data/interim`` root for cleaned outputs.

    Returns:
        GeometryCleanReport with per-region audit statistics.
    """
    catalog = Path(raw_catalog)
    interim = Path(interim_dir)
    forest_out = interim / "forest"
    forest_out.mkdir(parents=True, exist_ok=True)

    report = GeometryCleanReport()
    for region_id, _extract_dir, shp_path in find_forest_shapefiles(catalog):
        output_path = forest_out / f"{region_id}.gpkg"
        gdf = read_vector_without_autocorrect(shp_path)
        cleaned, stats = clean_geometries_split_policy(
            gdf,
            region_id=region_id,
            source_path=str(shp_path.resolve()),
            output_path=str(output_path.resolve()),
            read_without_ogr_autocorrect=True,
        )
        cleaned.to_file(output_path, driver="GPKG")
        report.records.append(stats)

    audit_path = interim / "clean_forest_geometry.json"
    audit_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return report


def prune_redundant_hydrography_extracts(
    raw_catalog: Path | str,
    *,
    preferred_format: str = "fgdb",
) -> list[str]:
    """Remove extracted hydrography folders for non-preferred formats (e.g. shp, lpk).

    Returns:
        List of deleted directory paths.
    """
    catalog = Path(raw_catalog)
    extract_root = catalog / "extracted"
    if not extract_root.is_dir():
        return []

    skip_suffixes = _redundant_hydro_suffixes(preferred_format)
    removed: list[str] = []
    for child in extract_root.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if not ("nbhn" in name and "rhnb" in name):
            continue
        if any(name.endswith(suffix) or suffix.strip("_") in name for suffix in skip_suffixes):
            _rm_tree(child)
            removed.append(str(child.resolve()))
    return removed


def _redundant_hydro_suffixes(preferred: str) -> tuple[str, ...]:
    """Return extract-dir name suffixes to prune given the preferred hydro format."""
    preferred = preferred.lower()
    if preferred == "fgdb":
        return ("_shp", "_lpk")
    if preferred == "shp":
        return ("_fgdb", "_lpk")
    if preferred == "lpk":
        return ("_fgdb", "_shp")
    return ()


def _rm_tree(path: Path) -> None:
    """Recursively delete a directory tree."""
    import shutil

    shutil.rmtree(path)


def hydro_extract_slug_for_zip(zip_path: Path) -> str:
    """Map a source zip path to its extract directory slug."""
    return _slugify(zip_path.stem)
