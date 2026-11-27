import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app, predictor
from app.model import rule_risk


class RuleRiskTests(unittest.TestCase):
    def test_high_risk_when_stock_is_above_two_times_expected_consumption(self) -> None:
        self.assertEqual(rule_risk(100, 10, 10, 30), "HIGH")

    def test_medium_risk_when_stock_is_above_expected_consumption(self) -> None:
        self.assertEqual(rule_risk(15, 10, 10, 30), "MEDIUM")

    def test_low_risk_when_stock_is_within_expected_consumption(self) -> None:
        self.assertEqual(rule_risk(10, 10, 10, 30), "LOW")

    def test_zero_days_with_stock_is_high_risk(self) -> None:
        self.assertEqual(rule_risk(1, 0, 0, 0), "HIGH")


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_predict_contract_uses_fallback_without_model(self) -> None:
        payload = {
            "currentStock": 120,
            "averageMonthlySales": 20,
            "daysUntilExpiry": 60,
            "recentSales": 18,
        }
        with patch.object(predictor, "model", None):
            response = self.client.post("/predict", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"risk": "HIGH", "fallback_used": True})

    def test_predict_accepts_snake_case_payload(self) -> None:
        payload = {
            "current_stock": 10,
            "average_monthly_sales": 10,
            "days_until_expiry": 30,
            "recent_sales": 10,
        }
        with patch.object(predictor, "model", None):
            response = self.client.post("/predict", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"risk": "LOW", "fallback_used": True})

    def test_negative_stock_is_rejected(self) -> None:
        payload = {
            "currentStock": -1,
            "averageMonthlySales": 20,
            "daysUntilExpiry": 60,
            "recentSales": 18,
        }
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_health_reports_model_state(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertIn("model_loaded", response.json())


if __name__ == "__main__":
    unittest.main()
