# Terra OBIA Stand Delineation Accuracy Report

## Summary

- **Overall accuracy (object-level):** 55.0%
- **Mean polygon IoU (cover type):** 0.0%

- **Model ID:** `stand_20260621T181026Z_23d8ae05`
- **Training data:** GeoNB harmonized labeled stands v1 (shape + inventory attrs); class_weight=balanced

## Cover type — precision / recall / F1

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| agriculture | 56.9% | 39.9% | 46.9% | 8016 |
| aquatic_bed | 24.9% | 81.1% | 38.1% | 417 |
| barren | 51.6% | 82.2% | 63.4% | 1912 |
| beach | 84.6% | 79.1% | 81.8% | 575 |
| bog | 58.6% | 66.9% | 62.5% | 2221 |
| coastal_marsh | 93.2% | 80.9% | 86.6% | 608 |
| conifer | 67.2% | 39.6% | 49.9% | 107301 |
| deciduous | 34.2% | 40.7% | 37.1% | 59367 |
| defense | 3.9% | 52.6% | 7.2% | 156 |
| developed | 74.7% | 52.3% | 61.5% | 11017 |
| dune | 30.8% | 45.1% | 36.6% | 82 |
| fen | 63.8% | 57.7% | 60.6% | 2494 |
| herbaceous | 21.7% | 87.3% | 34.8% | 393 |
| industrial | 4.8% | 35.4% | 8.5% | 1490 |
| infrastructure | 57.9% | 30.7% | 40.1% | 2103 |
| mixed | 27.7% | 30.2% | 28.9% | 45975 |
| recreational | 4.9% | 30.8% | 8.5% | 253 |
| rocky_shore | 30.1% | 46.8% | 36.7% | 47 |
| scrub | 7.6% | 94.6% | 14.0% | 148 |
| shrub | 53.0% | 86.4% | 65.7% | 992 |
| tidal_flat | 43.7% | 63.1% | 51.6% | 149 |
| water | 18.8% | 44.2% | 26.4% | 3460 |
| wetland_forest | 15.3% | 48.1% | 23.3% | 1129 |
| wetland_marsh | 30.9% | 15.8% | 20.9% | 3676 |
| wetland_shrub | 83.1% | 71.3% | 76.7% | 11984 |
| wetland_unknown | 1.6% | 61.1% | 3.1% | 357 |

## Canopy closure — precision / recall / F1

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| dense | 58.0% | 68.2% | 62.7% | 86288 |
| moderate | 60.4% | 55.6% | 57.9% | 88671 |
| open | 95.4% | 84.3% | 89.5% | 59734 |
| sparse | 82.6% | 79.3% | 80.9% | 31629 |

## Interpretation

Higher overall accuracy indicates stronger agreement between predicted
stand attributes and manual reference labels. Mean polygon IoU captures
spatial agreement of stand boundaries — the primary metric for comparing
against eCognition manual delineation.
