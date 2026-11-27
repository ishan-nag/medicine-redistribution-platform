# Project Log

## Monthly Plan (Point 6)
| Month | Planned work |
|---|---|
| Nov 2026 | Finalise scope, title, problem statement, stack, team roles, and GitHub repo. ML: define features, JSON contract, and labelling/fallback rule. |
| Dec 2026 | Core app (auth, inventory, expiry monitoring) and redistribution workflow. ML: generate synthetic dataset, compare Decision Tree vs Random Forest, build FastAPI `/predict` with fallback, and integrate with Spring Boot; target working endpoint by ~15 Dec. |
| Jan 2027 | Admin dashboard, analytics, integration testing, UI polish, deployment. ML: finish metrics, confusion matrix, feature importance chart, and model comparison for the report. |
| Feb 2027 | Final testing, documentation, report, and viva preparation; buffer if the project finishes early or slips. |

Ask the mentor whether the plan can end in Dec or Jan. If the project is done before 25 Dec, the later months remain buffer.

## Progress Log (Point 7) - newest first

### 2026-11-27 - dataset, model training, and model-backed API
- Done: reviewed and committed the 4000-row synthetic dataset; fixed the generator to use a script-relative output path and aligned `DATASET_NOTES.md` with the real parameters (Poisson λ=6, σ=0.3/0.7, 35%/30% flips). Regeneration produced an identical SHA256 hash.
- Done: added `app/features.py`; the service now derives `salesRate`, `expectedConsumption`, `stockToExpectedRatio`, and `remainingMonths` before model inference. Fixed NumPy numeric label normalisation so `np.int64` predictions are not mistaken for fallback cases.
- Done: added `train_model.py` with a stratified 80/20 split, seed 42, and Decision Tree vs Random Forest comparison. Decision Tree: accuracy 0.9050, F1 macro 0.9053. Random Forest: accuracy 0.9137, F1 macro 0.9141. Random Forest chosen.
- Done: generated `models/risk_model.pkl`, `models/model_metadata.json`, and reports (`model_comparison.md`, `metrics.json`, `confusion_matrix.png`, `feature_importance.png`). Added `.gitignore` exception so the model is deployable.
- Verified: 10/10 unit tests passed; live curl returned `{"status":"ok","model_loaded":true}` and `{"risk":"HIGH","fallback_used":false}`.
- Decisions: select by F1 macro then accuracy; use derived demand features; keep the model in the repo; store report artifacts under `ai-service/reports/`; keep the stack free/local.
- Blockers: none for the ML core. Spring Boot integration testing is pending until the backend endpoint call is ready.
- Next: integrate with Spring Boot, run deployment smoke test, prepare viva explanation from the comparison report and charts.

### 2026-11-27 - fallback-first FastAPI service
- Done: added `ai-service/app/main.py`, `app/model.py`, `app/schemas.py`, requirements files, `Procfile`, storage folders, and `tests/test_service.py`.
- Done: `/predict` accepts the backend camelCase JSON, accepts snake-case as a fallback, validates non-negative numbers, and returns `risk` plus `fallback_used`. `/health` reports `model_loaded`.
- Done: rule fallback implements the 0.7/0.3 blended sales rate, 2x HIGH threshold, 1x MEDIUM threshold; model loader tries `models/risk_model.pkl` and falls back if the file is missing, broken, or produces an unknown label.
- Verified: 8/8 stdlib unit tests passed; live curl smoke test returned `{"status":"ok","model_loaded":false}` and `{"risk":"HIGH","fallback_used":true}`.
- Decisions: training feature order fixed as `[currentStock, averageMonthlySales, recentSales, daysUntilExpiry]`; numeric label map fixed as `0=LOW`, `1=MEDIUM`, `2=HIGH`; no paid services or required test framework.
- Blockers: synthetic dataset is still unavailable, so no training, comparison, metrics, or charts yet. `.gitignore` still ignores `*.pkl`.
- Next: review dataset, add generator, compare Decision Tree vs Random Forest, save the better model, then test with Spring Boot.

### 2026-11-27 - ML context setup
- Done: `docs/ML_CONTEXT.md` updated with API contract, labelling/fallback rule, planned `ai-service/` structure, run command, free/deployment notes, and current status.
- Done: `docs/PROJECT_LOG.md` expanded with the monthly plan and this dated progress entry; older entries kept.
- Done: GitHub remote verified as `https://github.com/ishan-nag/medicine-redistribution-platform.git`.
- Decisions: use existing `docs/` files; blended sales rate `0.7*averageMonthlySales + 0.3*recentSales`; `HIGH` above 2x expected consumption, `MEDIUM` above 1x, otherwise `LOW`; always return `fallback_used`; keep the stack free/local and easy to deploy.
- Blockers: synthetic dataset is not available yet, so no dataset generation or model training has started. `.gitignore` ignores `*.pkl`; an exception is needed before committing the trained model.
- Next: review the synthetic dataset when provided, then build the dataset generator, model comparison, FastAPI endpoint, and fallback.

### 2026-11-27
- Project scope and title finalised
- GitHub repo created with backend / frontend / ai-service / docs / assets structure
