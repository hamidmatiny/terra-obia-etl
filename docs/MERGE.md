# Merging terra-obia-etl into terra-OBIA

This document describes how `terra-obia-etl` folds into the `terra-OBIA` monorepo
as a near-drop-in subpackage, and which upstream modules we depend on today.

## terra-OBIA modules depended on

| terra-OBIA module | Purpose in ETL | Merge target |
|-------------------|----------------|--------------|
| `terra_pipeline.tiling.grid.TileGrid` | Same 1024×1024 / 64 px overlap tiling for export | Keep import; no move |
| `terra_pipeline.models.RasterProfile`, `TileRecord` | Catalog-ready tile metadata | Keep import |
| `terra_core.io.cog.CogReader`, `TileWindow` | COG read contract for validation (stub today) | Keep import |
| `terra_pipeline.validation.*` | CRS/resolution checks on exported rasters | Keep import |

**Not** depended on (avoid pulling API/torch into ETL CI):

- `terra_api`, `torch`, `torchvision`, FastAPI stack

## Canonical CRS

terra-OBIA tests and API docs use **EPSG:32619** (NAD83 / UTM zone 19N) for New
Brunswick forestry layers. ETL harmonization targets this CRS unless terra-OBIA
defines a different training contract later.

## Proposed merge layout

```
terra-OBIA/
├── pipeline/
│   └── terra_pipeline/
│       └── ingestion/
│           └── geonb/          ← terra_etl/* moves here
│               ├── discover/
│               ├── ingest/
│               ├── clean/
│               ├── harmonize/
│               └── export/
├── configs/
│   └── etl/
│       └── geonb.yaml
└── tests/
    └── test_geonb_etl.py
```

### Namespace changes

| Current (terra-obia-etl) | After merge |
|--------------------------|-------------|
| `terra_etl.discover` | `terra_pipeline.ingestion.geonb.discover` |
| `terra_etl.cli` → `terra-etl` script | `terra-geonb-etl` or subcommand on pipeline CLI |
| `configs/geonb.yaml` | `configs/etl/geonb.yaml` |

### pyproject.toml merge steps

1. Move `terra_etl` package under `pipeline/terra_pipeline/ingestion/geonb/`.
2. Add ETL-only deps to root `pyproject.toml`: `pyyaml`, `openpyxl`, (future) `laspy`, `pdal`.
3. Remove path dependency on self; ETL imports `terra_pipeline` / `terra_core` directly.
4. Extend CI job to `mypy pipeline` + `pytest tests/test_geonb_etl.py`.
5. Add `[tool.poetry.scripts] terra-geonb-etl = "..."`.

### What stays separate

- Raw data paths (`source_dir`) remain operator-configured; never committed.
- `data/raw_catalog`, `data/interim`, `data/processed` stay gitignored.
- GeoNB-specific filename heuristics stay in `geonb/` — other provinces get
  sibling packages (`ingestion/bc/`, etc.).

## Export contract (to confirm as terra-OBIA matures)

Target outputs for training:

1. **Feature COGs** — multi-band or single-band rasters (LiDAR-derived CHM/intensity,
   or stacked inputs) as valid COGs with internal 512×512 tiles.
2. **Label layer** — vector GeoPackage or raster mask aligned to the same grid,
   with class encoding:
   - `0` = background/nodata (matches segmentation label raster convention)
   - thematic IDs for forest / non-forest / wetland (TBD in harmonize module)
3. **Tile catalog** — SQLite via `TileCatalog` referencing exported COG paths,
   built with `TileGrid.build_tile_records()`.

`CogReader` is still a stub in terra-OBIA; export writers should match
`RasterProfile` + STAC-like `TileRecord` fields already used by the pipeline.

## Dependency alignment

Pinned versions in `terra-obia-etl/pyproject.toml` use the same caret ranges as
terra-OBIA for overlapping packages. After `poetry lock`, diff lockfiles:

```bash
cd terra-obia-etl && poetry lock
# Compare geopandas, rasterio, shapely, pydantic, numpy, pyproj versions
```

Flag any drift before merge. ETL adds: `pyyaml`, `openpyxl`; future ingest adds
`laspy` / PDAL bindings (not in terra-OBIA today).

### pandas 2.x vs 3.0

| | terra-OBIA | terra-obia-etl (initial) | terra-obia-etl (now) |
|--|------------|--------------------------|----------------------|
| `pyproject.toml` | no direct pin (via geopandas) | `^2.0.0` | `^3.0.0` |
| `poetry.lock` | **3.0.3** | 2.3.3 | 3.0.3 after re-lock |

**Will 2.x vs 3.0 cause merge issues?** Unlikely for the code we write today —
terra-OBIA’s pandas usage is narrow (`read_csv`, `DataFrame` construction in
tests/fixtures, typing unions with GeoDataFrame). pandas 3.0’s breaking changes
(copy semantics, nullable dtype defaults, removed deprecated APIs) do not touch
those paths. geopandas 1.1.3 already resolves against pandas 3.0.3 in the
terra-OBIA lockfile.

**Recommendation:** target **`pandas ^3.0.0` now** in terra-obia-etl so both repos
share the same lock resolution before merge. This avoids a silent upgrade during
monorepo consolidation and keeps CI environments identical. Re-run
`poetry lock && poetry install` after changing the constraint.

No ETL-specific pandas 3.0 workarounds are expected; if ingest hits an edge case
(e.g. `read_csv` dtype inference on GeoNB exports), fix forward on 3.0 rather
than maintaining a 2.x pin.
