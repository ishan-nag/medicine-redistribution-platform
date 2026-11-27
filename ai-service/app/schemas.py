from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, allow_inf_nan=False)

    current_stock: float = Field(alias="currentStock", ge=0)
    average_monthly_sales: float = Field(alias="averageMonthlySales", ge=0)
    days_until_expiry: float = Field(alias="daysUntilExpiry", ge=0)
    recent_sales: float = Field(alias="recentSales", ge=0)


class PredictResponse(BaseModel):
    risk: Literal["LOW", "MEDIUM", "HIGH"]
    fallback_used: bool
