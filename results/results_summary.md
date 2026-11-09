# LinkLens results

All numbers are averages over 5 unseen synthetic test datasets (± = spread across the 5).
Weights and thresholds were chosen on a separate dev dataset only.

Chosen weights: text 0.2, identifiers 0.6, time 0.2 (time scale 90 days).

## Method comparison

| Method | Precision | Recall | F1 | Avg. precision (AP) | Recall on hard links* | False alarms on trap pairs** |
|---|---|---|---|---|---|---|
| LinkLens (weighted) | 0.62 ± 0.03 | 0.47 ± 0.03 | 0.53 ± 0.02 | 0.56 ± 0.02 | 0.19 ± 0.04 | 0.13 ± 0.03 |
| Baseline A (text only) | 0.18 ± 0.02 | 0.28 ± 0.02 | 0.22 ± 0.02 | 0.13 ± 0.02 | 0.27 ± 0.03 | 0.01 ± 0.02 |
| Baseline B (identifiers only) | 0.86 ± 0.03 | 0.34 ± 0.03 | 0.49 ± 0.03 | 0.30 ± 0.03 | 0.00 ± 0.00 | 1.00 ± 0.00 |

*Hard links = truly linked report pairs that share NO identifier.
**Trap pairs = unrelated report pairs that DO share an identifier (recycled or popular numbers).

## Ablation (remove one signal at a time, threshold re-tuned on dev)

| Variant | Precision | Recall | F1 |
|---|---|---|---|
| Full LinkLens | 0.62 ± 0.03 | 0.47 ± 0.03 | 0.53 ± 0.02 |
| without text similarity | 0.97 ± 0.01 | 0.34 ± 0.03 | 0.50 ± 0.04 |
| without identifiers | 0.31 ± 0.03 | 0.33 ± 0.02 | 0.32 ± 0.03 |
| without time | 0.66 ± 0.01 | 0.39 ± 0.03 | 0.49 ± 0.03 |
| without conflict penalties | 0.61 ± 0.03 | 0.47 ± 0.03 | 0.53 ± 0.02 |

## Speed (scoring every pair)

| Reports | Pairs | C (s) | NumPy (s) | Pure Python (s) | C vs Python | C vs NumPy |
|---|---|---|---|---|---|---|
| 392 | 76,636 | 0.002 | 0.006 | 0.13 | 60x | 2.7x |
| 783 | 306,153 | 0.008 | 0.025 | 0.40 | 49x | 3.1x |
| 1180 | 695,610 | 0.020 | 0.075 | 0.89 | 43x | 3.7x |
| 2006 | 2,011,015 | 0.048 | 0.196 | 2.57 | 53x | 4.1x |

Scores from C, NumPy and pure Python were checked to be identical (max difference in the speed table CSV).
