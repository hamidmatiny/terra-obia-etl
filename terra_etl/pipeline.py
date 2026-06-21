"""ETL pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from terra_etl.clean.forest import clean_forest_geometries, prune_redundant_hydrography_extracts
from terra_etl.clean.csv_validate import validate_regional_forest_csvs
from terra_etl.clean.geometry import GeometryCleanReport
from terra_etl.clean.csv_validate import CsvValidationReport
from terra_etl.config import PipelineConfig
from terra_etl.discover import run_discovery
from terra_etl.discover.scanner import DiscoveryManifest
from terra_etl.ingest.models import IngestReport
from terra_etl.ingest.zip import ingest_zips


@dataclass
class PipelineResult:
    """Outcome of a pipeline run."""

    manifest: DiscoveryManifest
    zip_ingest: IngestReport | None = None
    forest_clean: GeometryCleanReport | None = None
    csv_validation: CsvValidationReport | None = None
    pruned_hydro_dirs: list[str] | None = None


def run_pipeline(
    config: PipelineConfig,
    *,
    config_path: Path | str | None = None,
    project_root: Path | None = None,
    ingest_zip: bool = True,
    clean_forest: bool = True,
) -> PipelineResult:
    """Run discover and enabled ingest/clean stages."""
    root = project_root or Path.cwd()
    resolved = config.resolve_paths(root)

    manifest = run_discovery(config, config_path=config_path, project_root=root)
    result = PipelineResult(manifest=manifest)

    if ingest_zip:
        manifest_json = Path(resolved.paths.raw_catalog) / "manifest.json"
        result.zip_ingest = ingest_zips(
            manifest_json,
            resolved.paths.raw_catalog,
            hydrography_preferred_format=config.hydrography.preferred_format,
        )
        result.pruned_hydro_dirs = prune_redundant_hydrography_extracts(
            resolved.paths.raw_catalog,
            preferred_format=config.hydrography.preferred_format,
        )

    if clean_forest and ingest_zip:
        result.forest_clean = clean_forest_geometries(
            resolved.paths.raw_catalog,
            resolved.paths.interim,
        )

    if result.forest_clean is not None and result.forest_clean.passed:
        csv_paths = [e.path for e in manifest.included if e.extension == ".csv"]
        result.csv_validation = validate_regional_forest_csvs(
            csv_paths,
            resolved.paths.interim,
        )

    return result
