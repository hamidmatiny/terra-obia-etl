# Terra OBIA Stand Delineation Accuracy Report

## Summary

- **Overall accuracy (object-level):** 69.9%
- **Mean polygon IoU (cover type):** 0.0%

- **Model ID:** `stand_20260621T034627Z_0d59a3fa`
- **Training data:** GeoNB harmonized labeled stands v1 (shape + inventory attrs)

## Cover type — precision / recall / F1

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| agriculture | 63.1% | 58.7% | 60.8% | 8016 |
| aquatic_bed | 58.7% | 48.0% | 52.8% | 417 |
| barren | 75.2% | 72.8% | 74.0% | 1912 |
| beach | 73.0% | 14.6% | 24.3% | 575 |
| bog | 58.7% | 63.1% | 60.8% | 2221 |
| coastal_marsh | 93.3% | 16.0% | 27.2% | 608 |
| conifer | 49.3% | 38.0% | 42.9% | 6000 |
| deciduous | 39.7% | 40.6% | 40.1% | 6000 |
| defense | 0.0% | 0.0% | 0.0% | 156 |
| developed | 68.7% | 80.6% | 74.2% | 11016 |
| dune | 1.1% | 7.3% | 1.8% | 82 |
| fen | 62.5% | 60.0% | 61.3% | 2494 |
| herbaceous | 62.0% | 43.5% | 51.1% | 393 |
| industrial | 25.0% | 15.4% | 19.0% | 1490 |
| infrastructure | 65.8% | 33.6% | 44.5% | 2103 |
| mixed | 40.6% | 30.4% | 34.8% | 6000 |
| recreational | 53.6% | 5.9% | 10.7% | 253 |
| rocky_shore | 1.9% | 2.1% | 2.0% | 47 |
| scrub | 38.7% | 8.1% | 13.4% | 148 |
| shrub | 75.5% | 72.6% | 74.0% | 992 |
| tidal_flat | 7.3% | 29.5% | 11.7% | 149 |
| water | 39.7% | 86.3% | 54.4% | 3460 |
| wetland_forest | 0.0% | 0.0% | 0.0% | 1129 |
| wetland_marsh | 45.7% | 17.4% | 25.2% | 3676 |
| wetland_shrub | 74.4% | 95.3% | 83.5% | 11985 |
| wetland_unknown | 27.3% | 3.4% | 6.0% | 357 |

## Canopy closure — precision / recall / F1

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| dense | 58.7% | 69.8% | 63.8% | 7378 |
| moderate | 59.7% | 52.7% | 56.0% | 7343 |
| open | 84.6% | 91.4% | 87.8% | 29291 |
| sparse | 91.1% | 81.6% | 86.1% | 27667 |

## Interpretation

Higher overall accuracy indicates stronger agreement between predicted
stand attributes and manual reference labels. Mean polygon IoU captures
spatial agreement of stand boundaries — the primary metric for comparing
against eCognition manual delineation.
