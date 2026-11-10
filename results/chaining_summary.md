# Tiers and campaign chaining

Averages over the 5 unseen test datasets. The chaining threshold was chosen on dev data only.

## Are the tier labels honest?

| Tier | Suggested links per dataset | Share that are truly linked |
|---|---|---|
| strong | 208 | 99.4% |
| uncertain | 341 | 33.2% |
| weak | 78007 | 0.4% |

## Pair by pair vs chained into campaigns

| Method | Precision | Recall | F1 | Recall on hard links |
|---|---|---|---|---|
| Pairs scoring above the tuned threshold (0.33) | 0.62 | 0.47 | 0.53 | 0.19 |
| Campaigns (links chained at 0.40) | 0.99 | 0.45 | 0.62 | 0.17 |

Campaigns found per dataset: 39.0 (true gangs: 40). Campaigns containing a single true gang: 96%.

## What happens as the chaining threshold changes

| Threshold | Precision | Recall | F1 | Pure campaigns | Largest campaign |
|---|---|---|---|---|---|
| 0.30 | 0.03 | 0.96 | 0.05 | 5% | 207 |
| 0.33 | 0.12 | 0.80 | 0.21 | 15% | 72 |
| 0.35 | 0.50 | 0.64 | 0.56 | 39% | 19 |
| 0.37 | 0.82 | 0.52 | 0.63 | 75% | 12 |
| 0.38 | 0.91 | 0.48 | 0.62 | 85% | 10 |
| 0.39 | 0.94 | 0.46 | 0.62 | 90% | 10 |
| 0.40 | 0.99 | 0.45 | 0.62 | 96% | 8 |
| 0.42 | 0.99 | 0.45 | 0.62 | 97% | 8 |
| 0.45 | 1.00 | 0.43 | 0.60 | 97% | 8 |
| 0.50 | 1.00 | 0.37 | 0.54 | 99% | 8 |
| 0.55 | 1.00 | 0.28 | 0.43 | 99% | 7 |
| 0.60 | 1.00 | 0.16 | 0.27 | 99% | 5 |
| 0.70 | 0.99 | 0.06 | 0.11 | 99% | 3 |
