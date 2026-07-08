"""Command-line interface for terra-obia-etl."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from terra_etl.clean.geometry import GeometryCleanReport
from terra_etl.config import PipelineConfig
from terra_etl.discover import run_discovery
from terra_etl.harmonize import run_harmonize
from terra_etl.harmonize.harmonize import HarmonizeReport
from terra_etl.pipeline import run_pipeline


def _project_root() -> Path:
    """Return project root (directory containing pyproject.toml)."""
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return Path.cwd()


def _cmd_discover(args: argparse.Namespace) -> int:
    """Run discovery only and print manifest summary."""
    root = _project_root()
    config_path = Path(args.config).resolve()
    config = PipelineConfig.from_yaml(config_path).resolve_paths(root)
    manifest = run_discovery(config, config_path=config_path, project_root=root)

    print(f"Scanned: {manifest.total_scanned} candidate files under {manifest.source_dir}")
    print(f"Included: {manifest.included_count}  |  Ignored: {manifest.ignored_count}")
    print(f"Manifest: {root / config.paths.raw_catalog / 'manifest.json'}")

    if manifest.included:
        print("\n--- INCLUDED ---")
        for entry in manifest.included:
            print(f"  [{entry.layer_hint.value:12}] {entry.path}")
            print(f"               {entry.reason}")

    if manifest.ignored:
        print("\n--- IGNORED ---")
        for entry in manifest.ignored:
            print(f"  {entry.path}")
            print(f"    → {entry.reason}")

    return 0


def _print_geometry_clean_report(
    clean: GeometryCleanReport,
    audit_path: Path,
    title: str,
) -> None:
    """Print geometry clean summary for one layer group."""
    print(f"\n{title}: {'PASSED' if clean.passed else 'FAILED'}")
    print(f"Audit: {audit_path}")
    for record in clean.records:
        dist = record.area_change_distribution
        dist_summary = ""
        if dist is not None:
            dist_summary = (
                f" area_median={dist.median_pct:.3f}% p90={dist.p90_pct:.3f}% "
                f">{record.area_change_threshold:.0%}={record.area_change_count}"
            )
        print(
            f"  [{record.region_id}] in={record.total_features_in} out={record.total_features} "
            f"dropped_sliver={record.dropped_sliver_count} "
            f"fixed={record.fixed_count} (buffer0={record.repair_buffer0_count}, "
            f"exterior={record.repair_exterior_ring_count}) "
            f"shell_baseline={record.outer_shell_baseline_count}{dist_summary}"
        )


def _print_harmonize_report(
    report: HarmonizeReport,
    processed_dir: Path,
    *,
    audit_name: str = "harmonize_audit.json",
) -> None:
    """Print harmonize summary including class distributions and overlap audit."""
    print(f"\nHarmonize: {'PASSED' if report.passed else 'FAILED'}")
    print(f"Output GPKG: {report.output_gpkg}")
    print(f"Output CSV:  {report.output_csv}")
    print(f"Audit: {processed_dir / audit_name}")
    print(f"Features: {report.feature_count:,}")
    print("\n--- cover_type distribution ---")
    for label, count in sorted(report.cover_type_counts.items(), key=lambda x: -x[1]):
        print(f"  {label:22} {count:>10,}")
    print("\n--- canopy_closure_class distribution ---")
    for label, count in sorted(report.canopy_closure_counts.items(), key=lambda x: -x[1]):
        print(f"  {label:22} {count:>10,}")
    print("\n--- overlap audit (before clipping, area m²) ---")
    for pair, area in sorted(report.overlap_before.get("pair_areas_m2", {}).items()):
        print(f"  {pair:28} {area:>18,.0f}")
    print("\n--- overlap audit (after clipping, area m²) ---")
    for pair, area in sorted(report.overlap_after.get("pair_areas_m2", {}).items()):
        print(f"  {pair:28} {area:>18,.0f}")


def _cmd_harmonize(args: argparse.Namespace) -> int:
    """Run harmonize stage on cleaned interim layers."""
    root = _project_root()
    config_path = Path(args.config).resolve()
    config = PipelineConfig.from_yaml(config_path).resolve_paths(root)
    forest_regions = args.forest_region or None
    output_basename = "labeled_stands"
    if forest_regions and args.clip_to_forest_bounds:
        output_basename = f"labeled_stands_{'_'.join(forest_regions)}_dryrun"
    _combined, report = run_harmonize(
        config.paths.interim,
        config.paths.raw_catalog,
        config.paths.processed,
        target_crs_epsg=config.target_crs_epsg,
        forest_regions=forest_regions,
        clip_to_forest_bounds=args.clip_to_forest_bounds,
        output_basename=output_basename,
    )
    audit_path = Path(report.output_gpkg).with_name(f"{output_basename}_audit.json")
    _print_harmonize_report(report, audit_path.parent, audit_name=audit_path.name)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    """Run full pipeline (discover → ingest → export)."""
    root = _project_root()
    config_path = Path(args.config).resolve()
    config = PipelineConfig.from_yaml(config_path).resolve_paths(root)

    if config.discover.require_confirmation and not args.yes:
        manifest = run_discovery(config, config_path=config_path, project_root=root)
        print(f"Discovery complete: {manifest.included_count} files selected.")
        if manifest.included_count == 0:
            print("No files to ingest. Review config patterns or source_dir.")
            return 1
        print("\nReview the manifest before proceeding:")
        print(f"  {root / config.paths.raw_catalog / 'manifest.json'}")
        print("\nRe-run with --yes to proceed to ingest.")
        return 0

    result = run_pipeline(
        config,
        config_path=config_path,
        project_root=root,
        ingest_zip=True,
        harmonize=args.harmonize,
    )
    manifest = result.manifest
    print(f"Discovery complete: {manifest.included_count} files selected.")

    if manifest.included_count == 0:
        print("No files to ingest. Review config patterns or source_dir.")
        return 1

    if result.zip_ingest is None:
        print("No ingest stages ran.")
        return 1

    zip_report = result.zip_ingest
    print(f"\nZip ingest: {zip_report.ok_count} extracted, {zip_report.failed_count} failed")
    print(f"Log: {config.paths.raw_catalog / 'ingest_zip.json'}")

    for record in zip_report.records:
        status = record.status.value
        name = Path(record.source_path).name
        print(f"  [{status:7}] {name} → {record.output_dir or '—'}")
        if record.status.value != "ok" and record.status.value != "skipped":
            print(f"           {record.message}")
        elif record.member_count:
            print(f"           {record.member_count} files")

    if zip_report.failed_count:
        return 1

    if result.pruned_hydro_dirs:
        print(f"\nPruned redundant hydrography extracts: {len(result.pruned_hydro_dirs)}")
        for path in result.pruned_hydro_dirs:
            print(f"  removed {path}")

    if result.forest_clean is not None:
        _print_geometry_clean_report(
            result.forest_clean,
            config.paths.interim / "clean_forest_geometry.json",
            "Forest geometry clean",
        )
        if not result.forest_clean.passed:
            return 1

    if result.vector_clean is not None:
        for layer in ("non_forest", "wetland"):
            layer_records = [r for r in result.vector_clean.records if r.region_id == layer]
            if not layer_records:
                continue
            layer_report = GeometryCleanReport(records=layer_records)
            _print_geometry_clean_report(
                layer_report,
                config.paths.interim / f"clean_{layer}_geometry.json",
                f"{layer.replace('_', '-').title()} geometry clean",
            )
        if not result.vector_clean.passed:
            return 1

    if result.csv_validation is not None:
        val = result.csv_validation
        print(f"\nRegional forest CSV validation: {'PASSED' if val.passed else 'FAILED'}")
        print(f"Audit: {config.paths.interim / 'validate_forest_csv.json'}")
        for csv_record in val.records:
            status = "ok" if csv_record.passed else "FAIL"
            print(
                f"  [{status}] {Path(csv_record.csv_path).name} vs "
                f"{Path(csv_record.gpkg_path).name}: {csv_record.message}"
            )
        if not val.passed:
            return 1

    if result.vector_csv_validation is not None:
        val = result.vector_csv_validation
        print(f"\nNon-forest / wetland CSV validation: {'PASSED' if val.passed else 'FAILED'}")
        print(f"Audit: {config.paths.interim / 'validate_non_forest_wetland_csv.json'}")
        for csv_record in val.records:
            status = "ok" if csv_record.passed else "FAIL"
            print(
                f"  [{status}] {Path(csv_record.csv_path).name} vs "
                f"{Path(csv_record.gpkg_path).name}: {csv_record.message}"
            )
        if not val.passed:
            return 1

    if result.harmonize is not None:
        _print_harmonize_report(result.harmonize, config.paths.processed)
    elif args.harmonize:
        print("\nHarmonize skipped: vector clean did not complete successfully.")
        return 1
    else:
        print("\nNext stages: run `terra-etl harmonize --config ...` or re-run with --harmonize.")

    return 0


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="terra-etl",
        description="GeoNB ETL pipeline for terra-OBIA training data",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    discover_parser = sub.add_parser("discover", help="Scan source_dir and write manifest")
    discover_parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config (e.g. configs/geonb.yaml)",
    )

    run_parser = sub.add_parser("run", help="Run pipeline (discover → ingest → export)")
    run_parser.add_argument("--config", required=True, help="Path to YAML config")
    run_parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip manifest confirmation prompt",
    )

    run_parser.add_argument(
        "--harmonize",
        action="store_true",
        help="Run harmonize after clean/validation stages",
    )

    harmonize_parser = sub.add_parser(
        "harmonize",
        help="Harmonize interim layers to labeled training data",
    )
    harmonize_parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config (e.g. configs/geonb.yaml)",
    )
    harmonize_parser.add_argument(
        "--forest-region",
        action="append",
        dest="forest_region",
        metavar="REGION",
        help="Include only forest interim GPKG(s), e.g. r6_7 (repeatable)",
    )
    harmonize_parser.add_argument(
        "--clip-to-forest-bounds",
        action="store_true",
        help="Clip other layers to the forest subset bounds (dry-run helper)",
    )

    args = parser.parse_args(argv)
    if args.command == "discover":
        sys.exit(_cmd_discover(args))
    if args.command == "harmonize":
        sys.exit(_cmd_harmonize(args))
    if args.command == "run":
        sys.exit(_cmd_run(args))
    parser.print_help()
    sys.exit(2)


if __name__ == "__main__":
    main()
