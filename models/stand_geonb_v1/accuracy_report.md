# Terra OBIA Stand Delineation Accuracy Report

## Summary

- **Overall accuracy (object-level):** 61.3%
- **Mean polygon IoU (cover type):** 0.0%

- **Model ID:** `stand_20260621T060743Z_b5c69a67`
- **Training data:** GeoNB harmonized labeled stands v1 (shape + inventory attrs)

## Cover type — precision / recall / F1

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| agriculture | 61.6% | 60.3% | 61.0% | 8016 |
| aquatic_bed | 33.3% | 0.5% | 0.9% | 417 |
| barren | 68.2% | 74.6% | 71.3% | 1912 |
| beach | 2.1% | 2.8% | 2.4% | 575 |
| bog | 59.6% | 62.4% | 61.0% | 2221 |
| coastal_marsh | 0.3% | 0.3% | 0.3% | 608 |
| conifer | 52.7% | 92.7% | 67.2% | 107301 |
| deciduous | 38.1% | 17.4% | 23.9% | 59367 |
| defense | 0.0% | 0.0% | 0.0% | 156 |
| developed | 67.3% | 78.5% | 72.5% | 11017 |
| dune | 0.0% | 0.0% | 0.0% | 82 |
| fen | 62.8% | 61.9% | 62.4% | 2494 |
| herbaceous | 58.9% | 14.2% | 23.0% | 393 |
| industrial | 0.0% | 0.0% | 0.0% | 1490 |
| infrastructure | 50.8% | 35.6% | 41.9% | 2103 |
| mixed | 57.1% | 0.0% | 0.0% | 45975 |
| recreational | 7.5% | 1.6% | 2.6% | 253 |
| rocky_shore | 0.0% | 0.0% | 0.0% | 47 |
| scrub | 56.6% | 20.3% | 29.9% | 148 |
| shrub | 59.6% | 23.7% | 33.9% | 992 |
| tidal_flat | 0.0% | 0.0% | 0.0% | 149 |
| water | 49.0% | 31.8% | 38.6% | 3460 |
| wetland_forest | 0.0% | 0.0% | 0.0% | 1129 |
| wetland_marsh | 43.8% | 18.8% | 26.3% | 3676 |
| wetland_shrub | 74.7% | 95.6% | 83.9% | 11984 |
| wetland_unknown | 17.1% | 2.0% | 3.5% | 357 |

## Canopy closure — precision / recall / F1

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| dense | 58.0% | 68.1% | 62.7% | 86288 |
| moderate | 60.4% | 55.8% | 58.0% | 88671 |
| open | 92.1% | 89.3% | 90.7% | 59734 |
| sparse | 90.8% | 72.5% | 80.6% | 31629 |

## Interpretation

Higher overall accuracy indicates stronger agreement between predicted
stand attributes and manual reference labels. Mean polygon IoU captures
spatial agreement of stand boundaries — the primary metric for comparing
against eCognition manual delineation.
