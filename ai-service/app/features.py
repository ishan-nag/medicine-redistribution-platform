from __future__ import annotations

import numpy as np

RAW_FEATURE_ORDER = ("currentStock", "averageMonthlySales", "recentSales", "daysUntilExpiry")
MODEL_FEATURE_ORDER = RAW_FEATURE_ORDER + (
    "salesRate",
    "expectedConsumption",
    "stockToExpectedRatio",
    "remainingMonths",
)


def model_features(values) -> np.ndarray:
    data = np.asarray(values, dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != len(RAW_FEATURE_ORDER):
        raise ValueError(f"Expected {len(RAW_FEATURE_ORDER)} raw features")

    stock = data[:, 0]
    average_sales = data[:, 1]
    recent_sales = data[:, 2]
    days = data[:, 3]

    sales_rate = 0.7 * average_sales + 0.3 * recent_sales
    expected_consumption = sales_rate * (days / 30.0)
    stock_ratio = stock / (expected_consumption + 1e-6)
    remaining_months = days / 30.0

    return np.column_stack(
        [data, sales_rate, expected_consumption, stock_ratio, remaining_months]
    )
