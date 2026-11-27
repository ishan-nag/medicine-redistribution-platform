# ML Service - Context

## Purpose
Predicts expiry/wastage risk (LOW / MEDIUM / HIGH) for a medicine batch.

## Stack
Python, FastAPI, scikit-learn, pandas

## API contract (draft - confirm with backend)
POST /predict
Request:  {"currentStock":120,"averageMonthlySales":20,"daysUntilExpiry":60,"recentSales":18}
Response: {"risk":"HIGH"}

## Labelling rule
TBD

## Model
TBD (Decision Tree vs Random Forest)

## Folder structure
TBD

## How to run
TBD

## Status
Done: repo structure created
In progress: -
Pending: labelling rule, dataset generator, model, FastAPI endpoint, integration
