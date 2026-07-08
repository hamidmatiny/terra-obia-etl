# terra-obia-etl

Standalone ETL pipeline that ingests raw **GeoNB / GNB Open Data** (forest,
non-forest, wetland, and hydrography layers for New Brunswick) and produces
**harmonized, labeled stand polygons** for training classifiers in
[terra-OBIA](https://github.com/hamidmatiny/terra-OBIA).

v1 delivers **1.33M inventory-labeled polygons** (documented harmonize run; see
`docs/PROJECT_HISTORY.md` §4) with 26 `cover_type` classes and 4
`canopy_closure_class` bins, plus a trained gradient-boosting stand classifier
using shape metrics and provincial inventory attributes. There is **no raster
imagery** in the source data — pixel segmentation and COG export are deferred to
v2 when imagery is sourced.

**Full project narrative** (discovery pitfalls, nested-shell geometry repair,
harmonize performance, environment issues, training evaluation, v1 limitations):
→ **[docs/PROJECT_HISTORY.md](docs/PROJECT_HISTORY.md)**

Monorepo merge plan: **[docs/MERGE.md](docs/MERGE.md)**

## Quick start

```bash
poetry env use python3.11
poetry install

# 1. Scan Downloads (read-only) → review manifest before ingest
poetry run terra-etl discover --config configs/geonb.yaml

# 2. Ingest, clean geometry, validate CSV duplicates
poetry run terra-etl run --config configs/geonb.yaml --yes

# 3. Harmonize to labeled training GeoPackage / CSV (EPSG:32619)
poetry run terra-etl harmonize --config configs/geonb.yaml
```

Outputs land under `data/processed/` (e.g. `labeled_stands.gpkg`, `labeled_stands.csv`).

### Train the v1 stand classifier

Requires terra-OBIA as an editable optional dependency:

```bash
poetry install --extras terra-obia

poetry run python scripts/train_and_evaluate_stand_classifier.py \
  data/processed/labeled_stands.csv \
  models/stand_geonb_v1_balanced \
  0.2 100 \
  --class-weight balanced
```

Documented model artifacts: `models/stand_geonb_v1_balanced/` (balanced sample
weights — preferred over unweighted baseline for per-class equity).

## Layout

```
terra_etl/
  discover/    # scan source_dir, classify files, write manifest (review gate)
  ingest/      # extract zips, load shapefile / FGDB
  clean/       # geometry repair (nested-shell policy), CSV validation
  harmonize/   # CRS unify, label mapping, raster overlap filter, export
  cli.py
configs/geonb.yaml
scripts/train_and_evaluate_stand_classifier.py
data/          # working dirs (gitignored except .gitkeep)
docs/
  PROJECT_HISTORY.md
  MERGE.md
tests/
```

## Source data

Default `source_dir` is `~/Downloads` (set in `configs/geonb.yaml`). The pipeline
**never writes to source_dir** — only reads. Review `data/raw_catalog/manifest.csv`
after discover; ignored entries document redundant formats, Esri JSON exports, KMZ
preview subsamples, and out-of-scope LiDAR tiles.
