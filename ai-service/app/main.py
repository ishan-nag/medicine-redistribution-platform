from fastapi import FastAPI

from app.model import RiskPredictor
from app.schemas import PredictRequest, PredictResponse

app = FastAPI(title="Medicine Expiry Risk Service", version="0.1.0")
predictor = RiskPredictor()


@app.get("/health")
def health() -> dict[str, bool | str]:
    return {"status": "ok", "model_loaded": predictor.model_loaded}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    risk, fallback_used = predictor.predict(
        payload.current_stock,
        payload.average_monthly_sales,
        payload.recent_sales,
        payload.days_until_expiry,
    )
    return PredictResponse(risk=risk, fallback_used=fallback_used)
