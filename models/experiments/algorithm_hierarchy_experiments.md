# Stand classifier algorithm / hierarchy experiments

## Protocol (unchanged vs production train script)

- Data: `labeled_stands.csv` (1,331,610 rows)
- Features: `area_m2, perimeter_m, compactness, l1_ds, l1_sc, l1_vs, l1_pstock, lc_code, wri_code, spvc`
- Split: test_size=0.2, random_state=42, stratify=cover_type
- Holdout size: 266,322 rows (never used in `.fit()`)
- Metrics: cover accuracy, canopy accuracy, overall = mean of the two (same as `compute_object_classification_metrics` / accuracy_report.md)

## Results

| Experiment | Cover acc | Canopy acc | Overall (mean) | Cover macro F1 | Cover weighted F1 | Forest-only cover acc | Conifer recall | Deciduous recall | Mixed recall | Wall (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| existing_baseline_stand_geonb_v1 | 53.3% | 69.3% | 61.3% | 27.2% | 44.0% | 51.6% | 92.7% | 17.4% | 0.0% | 7 |
| existing_balanced_stand_geonb_v1_balanced | 41.1% | 69.0% | 55.0% | 41.2% | 44.2% | 37.9% | 39.6% | 40.7% | 30.2% | 7 |
| existing_stratified_on_its_own_holdout | 58.3% | 81.4% | 69.9% | 36.6% | 55.8% | 36.4% | 38.0% | 40.6% | 30.4% | 1 |
| existing_stratified_on_full_holdout | 43.4% | 69.1% | 56.3% | 35.8% | 45.1% | 37.7% | 38.8% | 41.0% | 30.7% | 4 |
| lightgbm_unweighted | 10.9% | 69.2% | 40.1% | 2.0% | 9.2% | 8.5% | 4.2% | 21.0% | 2.2% | 16 |
| lightgbm_balanced | 40.7% | 68.8% | 54.8% | 40.6% | 43.8% | 37.4% | 38.7% | 39.8% | 31.3% | 22 |
| xgboost_unweighted | 53.6% | 69.1% | 61.4% | 42.3% | 44.8% | 51.1% | 90.8% | 19.0% | 0.0% | 29 |
| xgboost_balanced | 39.9% | 68.8% | 54.4% | 40.7% | 43.0% | 36.4% | 36.6% | 38.8% | 33.0% | 29 |
| hierarchical_lgbm_unweighted | 54.0% | 69.2% | 61.6% | 42.9% | 44.8% | 51.7% | 92.6% | 17.7% | 0.0% | 12 |

## Notes per experiment

- **existing_baseline_stand_geonb_v1:** Committed artifact; trained on full CSV with same split seed (no class_weight)
  Top confusions: deciduous→conifer (48618), mixed→conifer (39717), conifer→deciduous (7604), mixed→deciduous (6035), agriculture→developed (2978)
- **existing_balanced_stand_geonb_v1_balanced:** Committed artifact; trained on full CSV with same split seed (class_weight=balanced)
  Top confusions: conifer→deciduous (32065), conifer→mixed (23003), mixed→deciduous (14407), deciduous→mixed (13186), deciduous→conifer (10573)
- **existing_stratified_on_its_own_holdout:** Committed stratified smoke artifact evaluated on holdout from the SAME capped 358k sample (optimistic; majority forest classes capped)
  Top confusions: agriculture→developed (3089), wetland_marsh→wetland_shrub (2901), conifer→deciduous (1863), mixed→deciduous (1841), developed→agriculture (1756)
- **existing_stratified_on_full_holdout:** Same stratified artifact scored on the FULL-province holdout (distribution shift vs its capped training sample)
  Top confusions: conifer→deciduous (32226), conifer→mixed (23693), mixed→deciduous (14531), deciduous→mixed (13211), deciduous→conifer (10340)
- **lightgbm_unweighted:** LightGBM n_estimators=100 max_depth=3 class_weight=None
  Top confusions: conifer→wetland_shrub (77462), deciduous→wetland_shrub (41282), mixed→wetland_shrub (33249), conifer→deciduous (21209), mixed→deciduous (8945)
- **lightgbm_balanced:** LightGBM n_estimators=100 max_depth=3 class_weight=balanced
  Top confusions: conifer→deciduous (31964), conifer→mixed (24129), mixed→deciduous (14262), deciduous→mixed (13781), deciduous→conifer (10461)
- **xgboost_unweighted:** XGBoost hist n_estimators=100 max_depth=3 balanced=False
  Top confusions: deciduous→conifer (47674), mixed→conifer (38534), conifer→deciduous (9633), mixed→deciduous (7246), agriculture→developed (3276)
- **xgboost_balanced:** XGBoost hist n_estimators=100 max_depth=3 balanced=True
  Top confusions: conifer→deciduous (32253), conifer→mixed (25656), deciduous→mixed (14509), mixed→deciduous (13919), deciduous→conifer (10161)
- **hierarchical_lgbm_unweighted:** Hierarchical LightGBM coarse→fine; coarse_acc_on_test=0.9738; cover_acc_on_true_forest_rows=0.516753431808242; groups=[np.str_('anthropogenic'), np.str_('forest'), np.str_('other'), np.str_('water'), np.str_('wetland')]
  Top confusions: deciduous→conifer (48602), mixed→conifer (39596), conifer→deciduous (7821), mixed→deciduous (6286), agriculture→developed (3422)

## Reading the numbers

- `existing_stratified_on_its_own_holdout` is the known optimistic smoke test (majority forest classes capped in the training sample).
- `existing_stratified_on_full_holdout` shows that artifact under the real province class mix.
- Algorithm swaps (LightGBM / XGBoost / hierarchy) use the **same** full-data holdout as `existing_baseline_*` / `existing_balanced_*`.
- `lightgbm_unweighted` is a real failure under these shared shallow hyperparameters (`n_estimators=100`, `max_depth=3`): it collapses toward `wetland_shrub` on forest rows. Balanced LightGBM recovers to ~balanced sklearn levels. This is reported, not discarded.

## Conclusion (honest)

With the **same features, same 80/20 stratified holdout, and no test-set tuning**:

| Comparison | Cover Δ vs baseline (53.3%) | Notes |
|---|---:|---|
| XGBoost unweighted | +0.3 pp | Essentially tied; same conifer-majority / mixed≈0 pattern |
| Hierarchical LGBM | +0.7 pp | Coarse group OA ~97%, but forest fine-class still ~51.7% |
| Balanced variants | −12 to −13 pp cover | Trade cover OA for more even forest recalls (as designed) |
| LightGBM unweighted | −42 pp | Unusable under these hyperparams |

**Nothing meaningfully improves cover-type accuracy without new features.** The bottleneck remains within-forest confusion (conifer ↔ deciduous ↔ mixed), which hierarchy does not fix once the coarse stage is already ~97% correct. Canopy stays ~69% across all full-data runs.

