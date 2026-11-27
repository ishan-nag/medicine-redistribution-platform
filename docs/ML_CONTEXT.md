# ML Service - Context

Last updated: 2026-11-27

This file is the current handoff context for Abhinandan and Claude Code. Update it after every milestone. Keep historical session notes in `docs/PROJECT_LOG.md`.

## Purpose
Predicts expiry/wastage risk (LOW / MEDIUM / HIGH) for a medicine batch.

Owner: Ishan (`ai-service/`).

## Stack
Python, FastAPI, scikit-learn, pandas.

No paid services or external ML APIs are required. The trained model runs locally; a deterministic rule fallback keeps `/predict` working if the model file or inference code fails.

## API contract (draft - confirm with Abhinandan/backend)
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

All four request fields are numbers.

Response:
```json
{
  "risk": "HIGH",
  "fallback_used": false
}
```

`risk` is one of `LOW`, `MEDIUM`, `HIGH`. `fallback_used` is `false` when the trained model made the prediction and `true` when the rule-based fallback was used. If the backend only needs `risk`, the extra field can be ignored safely.

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

The same rule is the fallback when the model is unavailable. The synthetic dataset will be generated from this rule later; add modest, realistic noise so the model cannot simply memorise a perfect formula. The exact noise level is still to be decided.

## Model
TBD until the synthetic dataset is available.

Plan: train and compare a Decision Tree and a Random Forest on the same train/test split. Choose the better model by held-out accuracy and F1, then save it at `ai-service/models/risk_model.pkl`.

Note: `.gitignore` currently ignores `*.pkl`. Add an exception for `ai-service/models/*.pkl` before committing the trained model, or deployment will not include it.

## Folder structure
Planned under `ai-service/`:

```text
ai-service/
  app/main.py
  app/model.py
  data/
  models/
  notebooks/
```

## How to run
Planned command, from `ai-service/` once the service exists:

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8001
```

For local integration, Spring Boot calls `http://localhost:8001/predict`. For deployment, use any free Python host that can run uvicorn; do not depend on a paid service.

## Status
Done:

- Repo structure and owners defined
- API contract drafted
- Labelling and fallback rule defined
- Context and progress log created under `docs/`
- GitHub remote verified

In progress:

- Waiting for the synthetic dataset before dataset generation and training

Pending:

- Synthetic dataset arrival/review
- Dataset generator
- Decision Tree vs Random Forest training, comparison, and saved model
- FastAPI `/predict` endpoint plus rule-based fallback
- Postman/curl test, then Spring Boot integration
- Metrics, confusion matrix, feature importance chart, and model comparison report
