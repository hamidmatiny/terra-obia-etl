# terra-obia-etl

Standalone ETL pipeline that ingests raw GeoNB / GNB Open Data (forest,
non-forest, wetland layers for New Brunswick) and produces training-ready
geospatial outputs for [terra-OBIA](https://github.com/hamidmatiny/terra-OBIA).

## Quick start

```bash
poetry install
poetry run terra-etl discover --config configs/geonb.yaml
```

Review the manifest under `data/raw_catalog/` before running ingest:

```bash
poetry run terra-etl run --config configs/geonb.yaml   # prompts for confirmation
poetry run terra-etl run --config configs/geonb.yaml --yes
```

## Layout

```
terra_etl/
  discover/   # scan source_dir, filter GeoNB-relevant files, write manifest
  ingest/     # format-specific readers (stubs → implemented incrementally)
  clean/      # validation, CRS, dedup
  harmonize/  # unify schemas across sources
  export/     # COG + label outputs for terra-OBIA
  cli.py
configs/      # YAML per data source
data/         # working dirs (gitignored except .gitkeep)
docs/MERGE.md # merge plan into terra-OBIA monorepo
tests/
```

## Source data

The default `source_dir` is `/Users/hamidrezamatiny/Downloads` (configurable in
YAML). The pipeline **never writes to source_dir** — only reads.
