# Model Comparison Report

## Dataset

- Rows: 4000
- Rule agreement with noisy labels: 93.27%
- Train/test split: 80%/20%, stratified, seed 42

## Comparison

| Model | Accuracy | F1 macro | F1 weighted |
|---|---:|---:|---:|
| Decision Tree | 0.9050 | 0.9053 | 0.9064 |
| Random Forest | 0.9137 | 0.9141 | 0.9154 |

## Chosen model: Random Forest

Selected by highest test F1 macro, then highest accuracy.

## Confusion matrix (chosen model)

| Actual / Predicted | LOW | MEDIUM | HIGH |
|---|---:|---:|---:|
| LOW | 283 | 33 | 0 |
| MEDIUM | 9 | 230 | 5 |
| HIGH | 0 | 22 | 218 |

## Feature importance (chosen model)

The model receives the four API fields plus derived demand features: currentStock, averageMonthlySales, recentSales, daysUntilExpiry, salesRate, expectedConsumption, stockToExpectedRatio, remainingMonths.

| Feature | Importance |
|---|---:|
| currentStock | 0.1242 |
| averageMonthlySales | 0.0110 |
| recentSales | 0.0122 |
| daysUntilExpiry | 0.0304 |
| salesRate | 0.0202 |
| expectedConsumption | 0.0635 |
| stockToExpectedRatio | 0.7083 |
| remainingMonths | 0.0302 |

## Interpretation

The dataset is synthetic and labels contain controlled noise near rule boundaries, so very high accuracy would suggest leakage or an overly memorised rule.

Charts: `confusion_matrix.png`, `feature_importance.png`. Raw metrics: `metrics.json`.
