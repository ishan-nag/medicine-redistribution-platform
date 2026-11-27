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
