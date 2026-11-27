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
  "fallback_used": true
}
```

`risk` is one of `LOW`, `MEDIUM`, `HIGH`. `fallback_used` is `false` when the trained model made the prediction and `true` when the rule-based fallback was used. Until the trained model exists, `/predict` always returns `fallback_used: true`. If the backend only needs `risk`, the extra field can be ignored safely.

GET /health
```json
{"status": "ok", "model_loaded": false}
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

The same rule is implemented as the fallback in `app/model.py`. The synthetic dataset will be generated from this rule later; add modest, realistic noise so the model cannot simply memorise a perfect formula. The exact noise level is still to be decided.

## Model
No model file yet. The service tries to load `ai-service/models/risk_model.pkl` at startup; if it is missing or broken, `/predict` keeps working through the rule fallback.

When training later, use this feature order:

```text
[currentStock, averageMonthlySales, recentSales, daysUntilExpiry]
```

Numeric predictions must be encoded as `0=LOW`, `1=MEDIUM`, `2=HIGH`; string labels are also accepted by the loader.

Plan: train and compare a Decision Tree and a Random Forest on the same train/test split. Choose the better model by held-out accuracy and F1, then save it at `ai-service/models/risk_model.pkl`.

Note: `.gitignore` currently ignores `*.pkl`. Add an exception for `ai-service/models/*.pkl` before committing the trained model, or deployment will not include it.

## Folder structure
Implemented under `ai-service/`:

```text
ai-service/
  app/main.py
  app/model.py
  app/schemas.py
  data/
  models/
  notebooks/
  tests/test_service.py
  requirements.txt
  requirements-dev.txt
  Procfile
```

## How to run
From `ai-service/`:

```bash
python -m pip install -r requirements-dev.txt
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
- 8 unit tests passing
- Live curl smoke test passed with fallback response
- Context and progress log created under `docs/`
- GitHub remote verified

In progress:

- Waiting for the synthetic dataset before dataset generation and training

Pending:

- Synthetic dataset arrival/review
- Dataset generator
- Decision Tree vs Random Forest training, comparison, and saved model
- Spring Boot integration testing
- Metrics, confusion matrix, feature importance chart, and model comparison report
