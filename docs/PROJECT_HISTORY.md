# terra-obia-etl — Project History & Engineering Narrative

This document records the full arc of **terra-obia-etl** from inception through the v1
stand-classifier milestone: what we built, what broke, how we diagnosed it, and why we
made the decisions we did. It is written for internal handoff and as source material for
portfolio or resume descriptions.

Monorepo merge plan: **[docs/MERGE.md](docs/MERGE.md)** — see also
**[docs/PROJECT_HISTORY.md](docs/PROJECT_HISTORY.md)** for the full engineering narrative.

---

## 1. Project origin & goal

### What terra-obia-etl is

**terra-obia-etl** is a standalone Python ETL pipeline that ingests raw **GeoNB / GNB Open
Data** forestry and wetland vector layers for New Brunswick and produces **harmonized,
labeled stand polygons** suitable for training classifiers in
[terra-OBIA](https://github.com/hamidmatiny/terra-OBIA).

It is **not** a pixel segmentation trainer. v1 stops at **tabular / object-level**
labels (geometry + inventory attributes + derived shape metrics) exported as GeoPackage
and CSV. The downstream consumer is terra-OBIA's `train_stand_classifier` — a gradient
boosting model on OBIA object features, not a U-Net on rasters.

### Relationship to terra-OBIA

| Repo | Role |
|------|------|
| **terra-OBIA** | Core library: tiling, COG I/O contracts, stand delineation classifier, API stubs |
| **terra-obia-etl** | Province-specific ingestion, cleaning, harmonization, label design |

We kept them **separate-but-mergeable** deliberately:

- GeoNB filename heuristics, nested-shell geometry repair, and NB CRS quirks belong in
  an ingestion package, not in core ML code.
- terra-OBIA's CI should not pull 10–20 GB of raw downloads or torch unless needed.
- [MERGE.md](./MERGE.md) describes folding `terra_etl/` into
  `terra_pipeline.ingestion.geonb/` when the contract stabilizes.

Training today runs from **terra-obia-etl's Poetry env** with an optional path dependency:

```bash
poetry install --extras terra-obia   # installs ../terra-OBIA in editable mode
```

That gives `import terra_core` without juggling two terminals — after we learned the hard
way that environment drift causes silent failures (see §5).

### Canonical CRS

Harmonized outputs target **EPSG:32619** (NAD83 / UTM zone 19N), matching terra-OBIA's
New Brunswick convention.

---

## 2. Data discovery & audit

### Source material

Operator downloads from GeoNB / GNB Open Data land in a read-only folder (default:
`~/Downloads`). The corpus is roughly **10–20 GB** across **8+ formats**: shapefile zips,
FileGDB zips, CSV, GeoJSON, KML/KMZ, GPKG, Esri REST JSON saved as `.txt`, LiDAR `.laz`,
and metadata spreadsheets.

**Critical scope finding:** there is **zero raster imagery** in the source bundle — no
Sentinel, no provincial ortho, no NAIP. LiDAR exists only as **five sample tiles**
(~5 km² total). That single fact drove v1 scope: polygon labels + inventory attributes
only; COG export and pixel segmentation deferred until imagery is sourced separately.

### Discover / manifest gate

We never write to `source_dir`. The first pipeline stage is **`terra-etl discover`**,
which recursively scans configured extensions, classifies each file, marks duplicates,
and writes:

- `data/raw_catalog/manifest.json`
- `data/raw_catalog/manifest.csv`

**Why a manifest?** GeoNB ships the same inventory in many redundant exports. Ingesting
everything blindly would waste disk, double-count features, and hide subsampled previews
as if they were authoritative. The manifest is a **human review gate** before
`terra-etl run --yes`.

### Format & redundancy findings

| Finding | Action |
|---------|--------|
| Duplicate GDB vs shapefile zips for the same forest region | Keep shapefile zip (or one canonical export); ignore duplicate GDB zip via content hash / member overlap |
| Esri REST Feature Layer JSON saved as `.txt` (multi-GB) | Reclassified from `metadata` → `esri_json_export`; ignored (Web Mercator, redundant with shapefile/CSV) |
| Province-scale forest CSV (~3 GB, embedded WKT) | Ignored — different vintage/coverage from regional shapefiles |
| Regional forest CSVs (r6_7, r1_2) | **Not ingested** — validation-only join against cleaned GPKG on `STDLAB` / row counts |
| Non-forest / wetland CSVs | Same — tabular duplicates confirmed via `OBJECTID` / key joins |
| GeoJSON / KML for non-forest & wetland | Ignored — redundant WGS84 web exports of shapefile zips already ingested |
| **KMZ "preview subsample"** | **Ignored** — ~1,220 features vs ~428k–483k in the authoritative shapefile for the same region; would have been a **silent catastrophic under-sample** if treated as forest inventory |
| Hydrography: FGDB vs shapefile vs LPK | **FGDB only** — single zip with full NBHN/RHNB layer model; dropped ~1.2 GB of redundant shp/lpk extracts |
| LiDAR `.laz` (5 tiles) | Ignored for v1 — sample coverage only; polygon pipeline does not require them |

The KMZ catch is worth emphasizing: discovery reason string explicitly records *"Google
Earth preview subsample (~1k features), not the full regional inventory."* Without the
manifest, a filename matching `Forestry_R_3_4_5` could have entered the pipeline at
0.3% of the true feature count.

### LiDAR decision

Five `.laz` tiles (~5 km²) were catalogued and excluded with reason
`lidar_v1_out_of_scope`. LiDAR may enrich v2 features (CHM, intensity) but is neither
province-wide nor required for v1's inventory-driven classifier.

---

## 3. Geometry cleaning — the nested-shell problem

This is the most important technical narrative in the project.

### Starting point: silent OGR autocorrect

Forest shapefiles arrived with **invalid polygon topology** (nested shells / bad ring
winding). Pyogrio/OGR can **auto-correct on read**, which makes invalid geometry
disappear from metrics without an audit trail.

We disabled silent correction (`OGR_ORGANIZE_POLYGONS=SKIP`) and implemented explicit
repair in `terra_etl/clean/geometry.py` with JSON audit logs per region.

### First approach: `make_valid` — looked fine, wasn't

Initial repair used Shapely `make_valid`. Post-repair validity checks passed (**100%
valid**). Then we added **area-change auditing** comparing planar area before vs after
repair on every fixed feature.

Results were alarming:

| Signal | Value |
|--------|-------|
| Aggregate area inflation on repaired subset | ~**14–15%** province-wide |
| Features flagged >1% area change | thousands per region (e.g. 8,853 / 3,591 / 7,529) |
| Share of "fixed" features with meaningful distortion | ~**98%** |

### Visual diagnosis: nested shells, not benign topology fixes

Before/after plots for worst offenders (saved under
`data/interim/forest/nested_shell_analysis/`) revealed a consistent pattern:

- **Before (invalid):** outer stand boundary + inner artifact ring render as a **solid
  filled** polygon. Reported area **sums both rings** → inflated footprint.
- **After `make_valid`:** inner ring becomes a **hole** → thin annulus. Area **drops**
  sharply (often +80–90% *apparent* change vs raw invalid area).

The outer ring **is** the real stand boundary. `make_valid` "fixes" topology by punching
a hole, not by recovering the intended footprint.

### STDLAB=0 slivers vs real stands

Of invalid features flagged for area change:

| Category | Share | Policy |
|----------|-------|--------|
| `STDLAB=0` degenerate slivers | ~**18%** | **Drop** — not inventory |
| Real stands (`STDLAB≠0`) | ~**82%** | **Repair** — cannot discard |

A single blanket strategy (drop all invalid, or `make_valid` everything) would either
lose real inventory or distort it.

### Approved repair policy: split by category

Implemented in `clean_geometries_split_policy()`:

1. **Drop** `STDLAB=0` slivers (forest only).
2. **`buffer(0)`** on remaining invalid polygons.
3. If area change still **>1%** vs baseline, **`exterior_ring_footprint()`** — reconstruct
   solid polygon from outer ring only, discarding nested inner artifacts.
4. Hard fail if any invalid geometry remains after repair.

Exterior-ring reconstruction reduced **worst-case** area deltas from ~**90%** to ~**46%**
on known offenders — still not zero, but geometrically aligned with "outer shell = stand."

### Outer-shell baseline correction to the audit

Comparing post-repair area to **raw invalid `polygon.area`** was the wrong baseline for
nested shells: raw area was already inflated by double-counting rings.

`audit_baseline_area()` uses the **exterior-ring footprint of the original invalid
geometry** when raw area exceeds outer-shell area. After that correction, **true stand
footprint distortion approached 0%** on the audited subset — the remaining large
percentages were measuring inflation removal, not stand shrinkage.

The same split policy (without sliver drop) was applied to **non-forest** and **wetland**
layers — same nested-shell pattern, no `STDLAB` sliver rule.

---

## 4. Harmonization — building the label scheme

### CRS heterogeneity

| Layer group | Source CRS | EPSG |
|-------------|-------------|------|
| Forest (3 regions) | WGS 84 / Pseudo-Mercator | 3857 |
| Non-forest | NAD83(CSRS) / NB Stereographic | 2953 |
| Wetland | NAD83(CSRS) / NB Stereographic | 2953 |
| **Harmonized output** | NAD83 / UTM 19N | **32619** |

All layers reproject to EPSG:32619 before union and export.

### Label mapping design (`terra_etl/harmonize/mapping.py`)

We mapped provincial codes into terra-OBIA's **`cover_type`** (26 classes) and
**`canopy_closure_class`** (4 classes):

| Source | Fields | Mapping |
|--------|--------|---------|
| Forest | `L1FUNA` (48 functional attributes) | Group SW/HW/MW → conifer / deciduous / mixed; dead timber (`DFDS`, `DEAD`) → conifer |
| Forest | `L1CC` crown closure 0–5 | 0→open, 1→sparse, 2–3→moderate, 4–5→dense |
| Non-forest | `PLU`, `LC` | Granular developed / agriculture / infrastructure / … — **not collapsed** |
| Wetland | `WC`, `WRI`, `SPVC` | Granular bog / fen / marsh / tidal_flat / … — **not collapsed** |

**Why granular?** Collapsing rare wetland or coastal classes would hide evaluation failure
and throw away provincial semantics the inventory already encodes. The cost is class
imbalance at training time — addressed later with sample weighting.

Shape metrics (`area_m2`, `perimeter_m`, `compactness`) and inventory passthrough fields
(`l1_ds`, `l1_sc`, `l1_vs`, `l1_pstock`, `lc_code`, `wri_code`, `spvc`) form the v1
feature vector for `train_stand_classifier`.

### Overlap / priority at province scale

Forest alone is **~1M+ polygons**. Wetland, non-forest, and hydrography water overlap
thematically. Initial design used **vector geometric clipping** (`geometry.difference()`
in a priority stack: water > wetland > forest > non_forest).

**What broke:** full-province harmonize **hung 30+ minutes** with no progress after the
overlap audit completed. Profiling showed the stall on
`clip_by_mask(forest, ~1M polygons)` — vector overlay at provincial scale.

**Fix:** rasterized overlap workflow in `terra_etl/harmonize/overlap.py`:

1. **100 m grid** overlap audit via `rasterio.features.rasterize` — pairwise overlap
   areas between layer groups.
2. **Priority filter** (not vector clip): drop features with **>5%** of their area
   overlapping a higher-priority mask.
3. **Chunked processing** (10k features) with per-layer progress logging.

Full province harmonize: **~7.4 minutes**, **1,331,610** labeled features exported to
`data/processed/labeled_stands.gpkg` / `.csv`.

**Drop analysis:** 16,391 features removed (~1.2%). Size-distribution audit confirmed
drops were **conflict-driven** (small overlaps with higher-priority layers), not random
large-stand deletion.

---

## 5. Environment instability — a recurring theme

Three distinct environment failures cost real wall-clock time during this project.

### Issue A: Python 3.14 empty venv

Poetry auto-selected system Python **3.14**, created a fresh venv, and ran entry points
**without** `poetry install` completing in that venv → `ModuleNotFoundError: yaml`,
`numpy`, etc.

**Fix:** `poetry env use python3.11`, `poetry install`, remove stale `py3.14` venv.

### Issue B: terra_core / terra-OBIA cross-project confusion

Training script imports `terra_core` from terra-OBIA. That package is an **optional extra**
(`poetry install --extras terra-obia`), not installed by default ETL-only setup.

**Fix:** One-time `--extras terra-obia` install; run all training from terra-obia-etl.

### Issue C: Agent sandbox vs local terminal venv divergence

Cursor agent shells sometimes resolved Poetry's cache to a **different virtualenv path**
than the user's terminal (`/var/folders/.../cursor-sandbox-cache/...` vs
`~/Library/Caches/pypoetry/...`). Diagnostics in the agent showed numpy working while
the user's terminal failed — opposite fixes applied to wrong environments.

**Lessons (apply before any long run):**

```bash
poetry env info          # confirm Python 3.11 + expected Path
poetry env list          # one active venv, no stale py3.14
poetry run python -c "import numpy, terra_core"
```

Do **not** trust redirected log files alone — background jobs that exit before flush
produce **0-byte logs** with no error. Prefer foreground runs or `PYTHONUNBUFFERED=1`
for smoke tests. `verbose=1` on `GradientBoostingClassifier` (added in terra-OBIA
`training.py`) prints per-stage progress on future training runs.

---

## 6. Model training & evaluation

### What `train_stand_classifier` actually is

A **GradientBoostingClassifier** pair (cover type + canopy closure) trained on **tabular
features** per labeled polygon — not pixel segmentation. v1 feature columns:

```text
area_m2, perimeter_m, compactness, l1_ds, l1_sc, l1_vs, l1_pstock, lc_code, wri_code, spvc
```

**COG / tile export skipped for v1** — no imagery exists to stack into training rasters.
See [MERGE.md](./MERGE.md) for the future export contract.

Training script: `scripts/train_and_evaluate_stand_classifier.py`  
Models: `models/stand_geonb_v1/` (baseline), `models/stand_geonb_v1_balanced/` (v1 documented).

### Smoke test was optimistic

| Run | Rows | Sample design | Overall accuracy |
|-----|------|---------------|------------------|
| Stratified smoke | 358,394 | Cap conifer/deciduous/mixed at 30k each; retain all rare classes | **69.9%** |
| Full province | 1,331,610 | Natural class frequencies | **61.3%** (baseline) |

Capping majority forest classes in the smoke sample **hid conifer-collapse** — the model
looked balanced because validation forest counts were artificially equalized.

### Class imbalance diagnosis (baseline, full scale)

Dominant failure mode: **predict conifer** for everything forest-like.

| Class | Baseline F1 | Baseline recall | Issue |
|-------|-------------|-----------------|-------|
| conifer | 67.2% | **92.7%** | Default bucket |
| deciduous | 23.9% | 17.4% | Collapsed |
| mixed | ~0% | ~0% | Collapsed |
| rocky_shore, dune, defense, tidal_flat | 0% | 0% | Ignored |

Canopy closure fared better (weighted F1 ~69%) — driven by `l1_vs` and inventory codes.

### Class-weighting experiment

`GradientBoostingClassifier` has **no `class_weight` parameter**. We added
`TrainingConfig.class_weight="balanced"` and pass **`sample_weight=compute_sample_weight("balanced", y)`**
into `.fit()` for both models (terra-OBIA `training.py`).

Full retrain on 1.33M rows → `models/stand_geonb_v1_balanced/` (~3h 18m wall; see runtime
note below). Results are summarized above: **wins and costs**, not a uniform upgrade.

### Baseline vs balanced — summary metrics

| Metric | Baseline | Balanced | Δ |
|--------|----------|----------|---|
| Overall accuracy | 61.29% | 55.04% | −6.25 pp |
| Cover weighted F1 | 43.99% | 44.21% | +0.22 pp |
| Cover **macro** F1 | 27.17% | **41.21%** | **+14.04 pp** |
| Canopy weighted F1 | 69.52% | 69.28% | −0.24 pp |
| Canopy macro F1 | 72.99% | 72.76% | −0.23 pp |

**Honest v1 conclusion:** balancing did **not** uniformly improve the model. Cover
**weighted F1 is essentially flat** (+0.2 pp) — the headline macro-F1 gain (+14 pp) is
real, but it reflects **reallocation of errors**, not a free win. The model stopped
defaulting to conifer and started reaching rare classes, but several mid-frequency and
common classes **regressed** in exchange. We chose the balanced artifact as v1 because
per-class equity and rare-class visibility matter more than raw accuracy for this
inventory — not because balancing "fixed" classification.

### Balancing reallocated errors — clear wins and real costs

Sample weighting penalizes ignoring minority classes. That worked for classes the
baseline never predicted — but the same redistribution **hurt** classes that had been
working reasonably well under majority-vote dynamics.

**Notable F1 regressions (baseline → balanced):**

| Class | F1 baseline | F1 balanced | Δ | What happened |
|-------|-------------|-------------|---|---------------|
| water | 38.6% | 26.4% | **−12.2 pp** | Recall up (+12.5 pp) but precision collapsed (−30.2 pp) — over-prediction bleed |
| scrub | 29.9% | 14.0% | **−15.8 pp** | Recall surged to 95% while precision fell to 8% — classic weighting over-shoot |
| agriculture | 61.0% | 46.9% | **−14.1 pp** | Recall −20.4 pp; confused with developed under reweighted forest/non-forest boundary |
| developed | 72.5% | 61.5% | **−11.0 pp** | Precision held but recall −26.2 pp — errors pushed toward agriculture and forest classes |
| wetland_shrub | 83.9% | 76.7% | −7.1 pp | Still strong, but recall dropped 24 pp as model spread predictions elsewhere |
| conifer | 67.2% | 49.9% | −17.3 pp | Intended trade: no longer a 93% recall default bucket |

Other classes with F1 drops include barren (−7.8 pp), wetland_marsh (−5.4 pp), and
wetland_unknown (flat F1 but precision cratered to 1.6%).

**Notable F1 gains (baseline → balanced):**

| Class | F1 baseline | F1 balanced | Δ | Notes |
|-------|-------------|-------------|---|-------|
| coastal_marsh | 0.3% | 86.6% | **+86.3 pp** | From effectively unlearned to usable |
| beach | 2.4% | 81.8% | **+79.4 pp** | Same pattern — coastal classes were invisible to baseline |
| tidal_flat | 0.0% | 51.6% | **+51.6 pp** | |
| dune | 0.0% | 36.6% | **+36.6 pp** | |
| rocky_shore | 0.0% | 36.7% | **+36.7 pp** | |
| mixed | ~0% | 28.9% | **+28.9 pp** | Recall from ~0% to 30% — major forest-class win |
| deciduous | 23.9% | 37.1% | +13.3 pp | Recall +23 pp |
| shrub | 33.9% | 65.7% | +31.8 pp | High recall (86%) but check precision trade |

Forest sub-types and coastal rarities improved; **water, scrub, agriculture, and
developed** — all classes the baseline handled passably — got worse. Defense improved
from 0% to 7% F1 (still barely functional). The balanced model is **better for audit and
rare-class coverage**, worse for some production metrics on common land-cover types.

### Full 26-class + 4-class comparison tables

See tables in §6.1 (reproduced from `extended_eval.json` for both model dirs).

### Mixed class — balancing helped, but overlap remains

| | Baseline | Balanced |
|---|----------|----------|
| mixed recall | ~0% | **30.2%** |
| mixed F1 | ~0% | **28.9%** |

`mixed_confusion_breakdown` (balanced validation, 45,975 mixed stands):

| Predicted as | Count |
|--------------|-------|
| deciduous | 14,407 |
| mixed | 13,895 |
| conifer | 10,162 |
| wetland_unknown | 3,215 |
| industrial | 2,274 |

Balancing recovered mixed from zero — so the baseline collapse was ** largely imbalance**,
not pure label noise. But ~70% of mixed stands still map elsewhere, consistent with
**genuine feature-space overlap** between forest types without spectral data. Documented
as **v1 limitation**, not a bug to fix with weighting alone.

### Runtime anomaly

| Run | Wall time |
|-----|-----------|
| Baseline full (1.33M, n_estimators=100) | ~**1 hour** (terminal; no mid-fit logging) |
| Balanced full | ~**3 h 18 min** |

Micro-benchmark on 80k rows showed **~1–2% overhead** from `sample_weight` alone — not
enough to explain 3× wall time. Likely causes: different tree structures under weighting,
26 trees per boosting stage for multiclass cover (2,600 trees total), system load, and
no visibility into which of two sequential fits was running. **Not pursued further.**

**Practical fix:** `verbose=1` default on `GradientBoostingClassifier` for future runs.

---

### 6.1 Model evaluation tables (validation set, 80/20 stratified on cover_type)

Validation support is identical between runs (same split seed). Δ = balanced − baseline
(percentage points).

#### Cover type — 26 classes

| Class | Prec base | Prec bal | Δ | Rec base | Rec bal | Δ | F1 base | F1 bal | Δ | Support |
|-------|-----------|----------|---|----------|---------|---|---------|--------|---|---------|
| conifer | 52.7% | 67.2% | +14.6 | 92.7% | 39.6% | −53.0 | 67.2% | 49.9% | −17.3 | 107,301 |
| deciduous | 38.1% | 34.2% | −4.0 | 17.4% | 40.7% | +23.3 | 23.9% | 37.1% | +13.3 | 59,367 |
| mixed | 57.1% | 27.7% | −29.4 | 0.0% | 30.2% | +30.2 | 0.0% | 28.9% | +28.9 | 45,975 |
| wetland_shrub | 74.7% | 83.1% | +8.3 | 95.6% | 71.3% | −24.3 | 83.9% | 76.7% | −7.1 | 11,984 |
| developed | 67.3% | 74.7% | +7.3 | 78.5% | 52.3% | −26.2 | 72.5% | 61.5% | −11.0 | 11,017 |
| agriculture | 61.6% | 56.9% | −4.7 | 60.3% | 39.9% | −20.4 | 61.0% | 46.9% | −14.1 | 8,016 |
| wetland_marsh | 43.8% | 30.9% | −12.8 | 18.8% | 15.8% | −3.0 | 26.3% | 20.9% | −5.4 | 3,676 |
| water | 49.0% | 18.8% | −30.2 | 31.8% | 44.2% | +12.5 | 38.6% | 26.4% | −12.2 | 3,460 |
| fen | 62.8% | 63.8% | +1.0 | 61.9% | 57.7% | −4.3 | 62.4% | 60.6% | −1.8 | 2,494 |
| bog | 59.6% | 58.6% | −0.9 | 62.4% | 66.9% | +4.5 | 61.0% | 62.5% | +1.5 | 2,221 |
| infrastructure | 50.8% | 57.9% | +7.1 | 35.6% | 30.7% | −4.9 | 41.9% | 40.1% | −1.7 | 2,103 |
| barren | 68.2% | 51.6% | −16.6 | 74.6% | 82.2% | +7.6 | 71.3% | 63.4% | −7.8 | 1,912 |
| industrial | 0.0% | 4.8% | +4.8 | 0.0% | 35.4% | +35.4 | 0.0% | 8.5% | +8.5 | 1,490 |
| wetland_forest | 0.0% | 15.3% | +15.3 | 0.0% | 48.1% | +48.1 | 0.0% | 23.3% | +23.3 | 1,129 |
| shrub | 59.6% | 53.0% | −6.6 | 23.7% | 86.4% | +62.7 | 33.9% | 65.7% | +31.8 | 992 |
| coastal_marsh | 0.3% | 93.2% | +92.8 | 0.3% | 80.9% | +80.6 | 0.3% | 86.6% | +86.3 | 608 |
| beach | 2.1% | 84.6% | +82.5 | 2.8% | 79.1% | +76.3 | 2.4% | 81.8% | +79.4 | 575 |
| aquatic_bed | 33.3% | 24.9% | −8.5 | 0.5% | 81.1% | +80.6 | 0.9% | 38.1% | +37.1 | 417 |
| herbaceous | 58.9% | 21.7% | −37.3 | 14.2% | 87.3% | +73.0 | 23.0% | 34.8% | +11.8 | 393 |
| wetland_unknown | 17.1% | 1.6% | −15.5 | 2.0% | 61.1% | +59.1 | 3.5% | 3.1% | −0.4 | 357 |
| recreational | 7.5% | 4.9% | −2.6 | 1.6% | 30.8% | +29.2 | 2.6% | 8.5% | +5.9 | 253 |
| defense | 0.0% | 3.9% | +3.9 | 0.0% | 52.6% | +52.6 | 0.0% | 7.2% | +7.2 | 156 |
| tidal_flat | 0.0% | 43.7% | +43.7 | 0.0% | 63.1% | +63.1 | 0.0% | 51.6% | +51.6 | 149 |
| scrub | 56.6% | 7.6% | −49.0 | 20.3% | 94.6% | +74.3 | 29.9% | 14.0% | −15.8 | 148 |
| dune | 0.0% | 30.8% | +30.8 | 0.0% | 45.1% | +45.1 | 0.0% | 36.6% | +36.6 | 82 |
| rocky_shore | 0.0% | 30.1% | +30.1 | 0.0% | 46.8% | +46.8 | 0.0% | 36.7% | +36.7 | 47 |

#### Canopy closure — 4 classes

| Class | Prec base | Prec bal | Δ | Rec base | Rec bal | Δ | F1 base | F1 bal | Δ | Support |
|-------|-----------|----------|---|----------|---------|---|---------|--------|---|---------|
| moderate | 60.4% | 60.4% | 0.0 | 55.8% | 55.6% | −0.1 | 58.0% | 57.9% | −0.1 | 88,671 |
| dense | 58.0% | 58.0% | 0.0 | 68.1% | 68.2% | +0.1 | 62.7% | 62.7% | 0.0 | 86,288 |
| open | 92.1% | 95.4% | +3.3 | 89.3% | 84.3% | −5.0 | 90.7% | 89.5% | −1.2 | 59,734 |
| sparse | 90.8% | 82.6% | −8.2 | 72.5% | 79.3% | +6.8 | 80.6% | 80.9% | +0.3 | 31,629 |

---

## 7. Current state & known limitations

### What works well (v1 balanced model)

- **Coastal / rare wetland classes** — coastal_marsh, beach, tidal_flat, dune, rocky_shore
  went from ~0% F1 to 37–87% under balancing (with precision/recall trade-offs on some).
- **Forest sub-type recovery** — mixed and deciduous recall improved materially vs
  baseline conifer-collapse; mixed still bleeds to conifer/deciduous (~70% misclassified).
- **Wetland_shrub, bog, fen** — remain usable (wetland_shrub F1 dipped 7 pp but still 77%).
- **Canopy open / sparse / dense / moderate** — ~69% weighted F1, largely unchanged by
  balancing; `l1_vs` + `wri_code` dominate importances.

### What does not work well

| Area | Symptom | Root cause |
|------|---------|------------|
| Forest sub-types | conifer ↔ deciduous ↔ mixed confusion | No spectral features; inventory overlap |
| Rare coastal (defense) | 7% F1, 53% recall, 4% precision | Tiny support + feature ambiguity |
| Scrub, wetland_unknown | High recall, terrible precision | Model over-predicts to satisfy weights |
| Water, agriculture, developed | **Regressed under balancing** vs baseline | Error reallocation — see §6 wins/costs table |

### v1 explicit non-goals

- Pixel segmentation / U-Net training
- COG or tile catalog export
- LiDAR feature derivation (province-wide)
- Any spectral or imagery input
- eCognition-style boundary IoU optimization (mean polygon IoU not computed in v1 eval)

### Documented v1 model choice

**`models/stand_geonb_v1_balanced/`** is the reference v1 artifact, not the unweighted
baseline — a **deliberate trade-off**, not a claim that balancing improved every class.
We prefer it for inventory work because macro F1 and rare-class recall improved sharply,
even though water, scrub, agriculture, and developed regressed. Use the baseline model
if headline accuracy on common classes matters more than coastal/forest sub-type equity.

Artifacts:

```text
models/stand_geonb_v1_balanced/
  cover_type_model.joblib
  canopy_closure_model.joblib
  metadata.json
  extended_eval.json      # per-class metrics, mixed breakdown, confusions, importances
  accuracy_report.md
```

Harmonized training data:

```text
data/processed/labeled_stands.gpkg
data/processed/labeled_stands.csv
```

---

## 8. Path to v2

### Spectral / imagery (highest leverage)

Source **Sentinel-2** (wide coverage, free) vs **provincial ortho** ( higher resolution,
access constraints). Once rasters exist, harmonized polygons become **labels** for tile
export per [MERGE.md](./MERGE.md) — COG features + aligned mask rasters + `TileGrid`
catalog.

### LiDAR

Integrate as **enrichment** (CHM, intensity stats zonal to stand polygons) when wider
coverage is available — not a v1 blocker.

### COG / tile export + pixel path

Unblocks terra-OBIA segmentation workflow (`CogReader`, training tiles). Depends on
imagery sourcing above.

### Mixed forest resolution

With spectral bands, expect conifer/deciduous/mixed separability to improve; may still
need hierarchical classification (forest vs non-forest first, then SW/HW/MW split) or
explicit mixed class modeling with spectral indices (NDVI, NBR, texture).

### Training ops

- Default `verbose=1` on GBM fits
- Consider `HistGradientBoostingClassifier` for faster large-n multiclass
- Resampling (`conifer` cap + rare-class upsample) as alternative to sample weights if
  runtime or precision/recall trade-offs need tuning

---

## Appendix: Key commands (current pipeline)

```bash
# Environment (once per machine)
poetry env use python3.11
poetry install
poetry install --extras terra-obia   # training only

# ETL
poetry run terra-etl discover --config configs/geonb.yaml
poetry run terra-etl run --config configs/geonb.yaml --yes
poetry run terra-etl harmonize --config configs/geonb.yaml

# Train documented v1 model
poetry run python scripts/train_and_evaluate_stand_classifier.py \
  data/processed/labeled_stands.csv \
  models/stand_geonb_v1_balanced \
  0.2 100 \
  --class-weight balanced
```

Dry-run harmonize on a single forest region:

```bash
poetry run terra-etl harmonize --config configs/geonb.yaml \
  --forest-region r6_7 --clip-to-forest-bounds
```
