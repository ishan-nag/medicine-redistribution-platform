from __future__ import annotations

import os
from pathlib import Path
from numbers import Real
from typing import Any

from app.features import model_features

RISK_LABELS = ("LOW", "MEDIUM", "HIGH")
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "risk_model.pkl"


def rule_risk(
    current_stock: float,
    average_monthly_sales: float,
    recent_sales: float,
    days_until_expiry: float,
) -> str:
    values = (current_stock, average_monthly_sales, recent_sales, days_until_expiry)
    if any(value < 0 for value in values):
        raise ValueError("Inventory values cannot be negative")

    sales_rate = 0.7 * average_monthly_sales + 0.3 * recent_sales
    expected_consumption = sales_rate * (days_until_expiry / 30)

    if current_stock > 2 * expected_consumption:
        return "HIGH"
    if current_stock > expected_consumption:
        return "MEDIUM"
    return "LOW"


def normalise_prediction(value: Any) -> str | None:
    if isinstance(value, str):
        label = value.strip().upper()
        if label in RISK_LABELS:
            return label
    if isinstance(value, Real) and not isinstance(value, bool):
        index = int(value)
        if 0 <= index < len(RISK_LABELS):
            return RISK_LABELS[index]
    return None


class RiskPredictor:
    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path or Path(os.getenv("RISK_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
        self.model = self._load_model()

    @property
    def model_loaded(self) -> bool:
        return self.model is not None

    def _load_model(self) -> Any | None:
        if not self.model_path.exists():
            return None
        try:
            import joblib

            candidate = joblib.load(self.model_path)
        except Exception:
            return None
        if hasattr(candidate, "predict"):
            return candidate
        return None

    def predict(
        self,
        current_stock: float,
        average_monthly_sales: float,
        recent_sales: float,
        days_until_expiry: float,
    ) -> tuple[str, bool]:
        fallback = rule_risk(
            current_stock,
            average_monthly_sales,
            recent_sales,
            days_until_expiry,
        )

        if self.model is None:
            return fallback, True

        features = model_features(
            [[current_stock, average_monthly_sales, recent_sales, days_until_expiry]]
        )
        try:
            raw_prediction = self.model.predict(features)
            label = normalise_prediction(raw_prediction[0])
        except Exception:
            return fallback, True

        if label is None:
            return fallback, True
        return label, False
