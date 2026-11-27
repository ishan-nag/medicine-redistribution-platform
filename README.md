# Medicine Expiry & Inter-Pharmacy Redistribution Platform

A platform that tracks batch-level pharmacy inventory, identifies medicines at risk of expiring unsold, and helps redistribute eligible stock between verified pharmacies.

## Repository structure
| Folder | Owner | Tech |
|---|---|---|
| `backend/` | Abhinandan | Java, Spring Boot, JWT, MySQL/PostgreSQL |
| `frontend/` | <name> | React |
| `ai-service/` | Ishan | Python, FastAPI, scikit-learn |
| `docs/` | All | Project log, context files, report |
| `assets/` | All | Images, diagrams, sample data |

## Team
- <Team ID>
- Mentor: <name>

## How the parts connect
React (UI) -> Spring Boot (business logic) -> FastAPI ML service (risk prediction: LOW / MEDIUM / HIGH)

## Rules
- Work only in your own folder.
- `git pull` before starting, commit small and often.
- Never commit `.env` files or API keys.
