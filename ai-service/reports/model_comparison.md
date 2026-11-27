# Model Comparison Report

## Dataset

- Rows: 4000
- Rule agreement with noisy labels: 93.27%
- Train/test split: 80%/20%, stratified, seed 42
- Hyperparameter search: 3-fold stratified CV, macro F1 scoring

## Models tested

Decision Tree, Random Forest, Extra Trees, Gradient Boosting, Hist Gradient Boosting, Logistic Regression

## Comparison

| Model | CV F1 macro | Accuracy | F1 macro | F1 weighted |
|---|---:|---:|---:|---:|
| Decision Tree | 0.9199 | 0.9125 | 0.9127 | 0.9141 |
| Random Forest | 0.9270 | 0.9213 | 0.9218 | 0.9229 |
| Extra Trees | 0.9014 | 0.9038 | 0.9025 | 0.9037 |
| Gradient Boosting | 0.9232 | 0.9075 | 0.9079 | 0.9090 |
| Hist Gradient Boosting | 0.9154 | 0.9087 | 0.9081 | 0.9096 |
| Logistic Regression | 0.9090 | 0.8962 | 0.8943 | 0.8961 |

## Chosen model: Random Forest

- Best parameters: max_depth=12, min_samples_leaf=3
- Test accuracy: 0.9213
- Test macro F1: 0.9218
- Selection rule: highest test F1 macro, then highest accuracy, then CV F1 macro.

## Confusion matrix (chosen model)

| Actual / Predicted | LOW | MEDIUM | HIGH |
|---|---:|---:|---:|
| LOW | 284 | 32 | 0 |
| MEDIUM | 7 | 235 | 2 |
| HIGH | 0 | 22 | 218 |

## Feature importance (chosen model)

Source: native.
The model receives the four API fields plus derived demand features: currentStock, averageMonthlySales, recentSales, daysUntilExpiry, salesRate, expectedConsumption, stockToExpectedRatio, remainingMonths.

| Feature | Importance |
|---|---:|
| currentStock | 0.1275 |
| averageMonthlySales | 0.0089 |
| recentSales | 0.0094 |
| daysUntilExpiry | 0.0276 |
| salesRate | 0.0169 |
| expectedConsumption | 0.0584 |
| stockToExpectedRatio | 0.7248 |
| remainingMonths | 0.0265 |

## Interpretation

The data is synthetic and labels include controlled boundary noise, so accuracy should stay below the 93.27% rule agreement. A near-perfect score would suggest leakage or a memorised rule.

## Artifacts

- `metrics.json`: full metrics and parameters for every model
- `model_comparison.csv`: spreadsheet-friendly comparison
- `hyperparameter_comparison.md`: grids, best parameters, CV and test scores
- `confusion_matrices.md`: confusion matrix for every model
- `confusion_matrix_<model>.png`: confusion matrix chart for every model
- `feature_importance_<model>.png`: importance chart for every model
- `confusion_matrix.png` and `feature_importance.png`: chosen model charts
