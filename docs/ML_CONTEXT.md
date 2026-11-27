# ML Service - Context

Last updated: 2026-11-27

This file is the current handoff context for Abhinandan and Claude Code. Update it after every milestone. Keep historical session notes in `docs/PROJECT_LOG.md`.

## Purpose
Predicts expiry/wastage risk (LOW / MEDIUM / HIGH) for a medicine batch.

Owner: Ishan (`ai-service/`).

## Stack
Python, FastAPI, scikit-learn, pandas.

No paid services or external ML APIs are required. The trained model runs locally; a deterministic rule fallback keeps `/predict` working if the model file or inference code fails.

## API contract (implemented - confirm field names with Abhinandan/backend)
POST /predict

Request:
```json
{
  "currentStock": 120,
  "averageMonthlySales": 20,
  "daysUntilExpiry": 60,
  "recentSales": 18
}
```

All four request fields are numbers and must be non-negative. Snake-case field names are also accepted.

Response:
```json
{
  "risk": "HIGH",
  "fallback_used": false
}
```

`risk` is one of `LOW`, `MEDIUM`, `HIGH`. `fallback_used` is `false` when the trained model made the prediction and `true` when the rule-based fallback was used. If the backend only needs `risk`, the extra field can be ignored safely.

GET /health
```json
{"status": "ok", "model_loaded": true}
```

## Labelling rule
Expected consumption = sales rate x remaining months:

```text
salesRate = 0.7 * averageMonthlySales + 0.3 * recentSales
expectedConsumption = salesRate * (daysUntilExpiry / 30)
```

Labels:

- `HIGH` if `currentStock > 2 * expectedConsumption`
- `MEDIUM` if `expectedConsumption < currentStock <= 2 * expectedConsumption`
- `LOW` if `currentStock <= expectedConsumption`

The same rule is implemented as the fallback in `app/model.py`. The synthetic dataset was generated from this rule with bounded boundary noise: 269/4000 rows (6.7%) were flipped, giving 93.27% rule agreement so the model cannot simply memorise a perfect formula.

## Model
Chosen: **Random Forest**, saved at `ai-service/models/risk_model.pkl`. Metadata is at `ai-service/models/model_metadata.json`.

Training script: `ai-service/train_model.py` uses an 80/20 stratified split with seed 42 and 3-fold stratified grid search scored by macro F1.

| Model | CV F1 macro | Accuracy | F1 macro | F1 weighted |
|---|---:|---:|---:|---:|
| Decision Tree | 0.9199 | 0.9125 | 0.9127 | 0.9141 |
| Random Forest | 0.9270 | 0.9213 | 0.9218 | 0.9229 |
| Extra Trees | 0.9014 | 0.9038 | 0.9025 | 0.9037 |
| Gradient Boosting | 0.9232 | 0.9075 | 0.9079 | 0.9090 |
| Hist Gradient Boosting | 0.9154 | 0.9087 | 0.9081 | 0.9096 |
| Logistic Regression | 0.9090 | 0.8962 | 0.8943 | 0.8961 |

Best parameters: `max_depth=12`, `min_samples_leaf=3`. Reason: Random Forest had the highest held-out F1 macro and accuracy, and the highest CV F1. Accuracy is below the 93.27% rule agreement because the labels contain boundary noise; this is expected and avoids a suspicious 99% result.

Raw API feature order:

```text
[currentStock, averageMonthlySales, recentSales, daysUntilExpiry]
```

Model feature order, built by `app/features.py`:

```text
[currentStock, averageMonthlySales, recentSales, daysUntilExpiry, salesRate, expectedConsumption, stockToExpectedRatio, remainingMonths]
```

Numeric labels are encoded as `0=LOW`, `1=MEDIUM`, `2=HIGH`; the loader accepts Python and NumPy numerics plus string labels.

Report artifacts live in `ai-service/reports/`: `REPORTS_INDEX.md`, `model_comparison.md`, `metrics.json`, `model_comparison.csv`, `hyperparameter_comparison.md`, `confusion_matrices.md`, per-model confusion matrices, and per-model feature importance charts. `.gitignore` permits `ai-service/models/*.pkl` so deployment includes the model.

## Folder structure
Implemented under `ai-service/`:

```text
ai-service/
  app/main.py
  app/model.py
  app/features.py
  app/schemas.py
  data/generate_dataset.py
  data/DATASET_NOTES.md
  data/synthetic_inventory_risk.csv
  models/risk_model.pkl
  models/model_metadata.json
  notebooks/
  reports/REPORTS_INDEX.md
  reports/model_comparison.md
  reports/model_comparison.csv
  reports/metrics.json
  reports/hyperparameter_comparison.md
  reports/confusion_matrices.md
  reports/confusion_matrix_<model>.png
  reports/feature_importance_<model>.png
  reports/confusion_matrix.png
  reports/feature_importance.png
  tests/test_service.py
  train_model.py
  requirements.txt
  requirements-dev.txt
  Procfile
```

## How to run
From `ai-service/`:

```bash
python -m pip install -r requirements-dev.txt
python data/generate_dataset.py
python train_model.py
python -m uvicorn app.main:app --host 0.0.0.0 --port 8001
python -m unittest discover -s tests -v
```

For local integration, Spring Boot calls `http://localhost:8001/predict`. For deployment, use `Procfile` with `requirements.txt` on any free Python host that can run uvicorn; do not depend on a paid service.

## Status
Done:

- Repo structure and owners defined
- API contract implemented with camelCase and snake-case input support
- Labelling and fallback rule implemented
- `/health` and `/predict` endpoints implemented
- Synthetic 4000-row dataset generated, documented, and verified reproducible
- Six-model comparison with hyperparameter grids completed
- Random Forest saved at `models/risk_model.pkl` with metadata
- Full metrics, per-model confusion matrices, per-model feature importance charts, and comparison reports generated
- 10 unit tests passing
- Live curl smoke test passed with `model_loaded: true` and `fallback_used: false`
- Context and progress log created under `docs/`
- GitHub remote verified

In progress:

- Spring Boot integration testing with Abhinandan

Pending:

- Final deployment smoke test after backend integration
- Viva explanation using `ai-service/reports/` artifacts
