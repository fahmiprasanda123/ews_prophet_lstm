"""
Pydantic schemas for the Agri-AI EWS REST API.
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import date


class PriceRecord(BaseModel):
    date: str
    province: str
    commodity: str
    price: float


class ForecastRequest(BaseModel):
    province: str = Field(..., example="DKI Jakarta")
    commodity: str = Field(..., example="Beras")
    days: int = Field(30, ge=1, le=120, description="Forecast horizon in days")
    model: str = Field("hybrid", description="Model: prophet, lstm, tft, hybrid")


class ForecastPoint(BaseModel):
    date: str
    predicted_price: float
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None


class NarrativeFactor(BaseModel):
    name: str
    impact: str = Field(..., description="Impact level: high, medium, low")
    description: str


class PriceNarrative(BaseModel):
    direction: str = Field(..., description="NAIK, TURUN, or STABIL")
    pct_change: float
    summary: str
    factors: List[NarrativeFactor]
    narrative: str


class ForecastResponse(BaseModel):
    province: str
    commodity: str
    model_used: str
    current_price: float
    forecast: List[ForecastPoint]
    metrics: Optional[dict] = None
    ews_status: Optional[dict] = None
    narrative: Optional[PriceNarrative] = None


class EWSAlert(BaseModel):
    province: str
    commodity: str
    level: str
    score: float
    message: str
    factors: dict
    recommendations: List[str]


class EWSStatusResponse(BaseModel):
    alerts: List[EWSAlert]
    timestamp: str
    total_danger: int = 0
    total_alert: int = 0


class ModelMetrics(BaseModel):
    model_name: str
    rmse: float
    mae: float
    mape: float
    r2: Optional[float] = None
    smape: Optional[float] = None
    directional_accuracy: Optional[float] = None


class ModelComparisonResponse(BaseModel):
    province: str
    commodity: str
    models: List[ModelMetrics]
    best_model: str


class SupplyRiskResponse(BaseModel):
    province: str
    commodity: str
    score: float
    factors: dict
    trend_direction: str
    description: str


class DataStatsResponse(BaseModel):
    total_records: int
    provinces: int
    commodities: int
    date_from: Optional[str] = None
    date_to: Optional[str] = None


class SyncResponse(BaseModel):
    status: str
    records_added: int
    message: str
