"""Pydantic response/validation schemas for the forecast API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

ASSET_PATTERN = r"^[A-Za-z0-9]{2,12}$"
MODEL_PATTERN = r"^[a-z0-9_]{2,20}$"


class RiskLevel(BaseModel):
    """VaR and ES at one confidence level, as positive 1-day losses in percent."""

    confidence: float = Field(..., examples=[0.99])
    var_pct: float = Field(..., description="1-day Value-at-Risk, % loss", examples=[2.31])
    es_pct: float = Field(..., description="1-day Expected Shortfall, % loss", examples=[2.98])


class ForecastResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    asset: str
    asset_name: str
    model: str
    model_label: str
    last_close_date: str = Field(..., description="last daily close used by the forecast")
    forecast_for: str = Field(..., description="trading day the forecast applies to")
    vol_1d_pct: float | None = Field(None, description="next-day volatility (daily st.dev., %)")
    vol_5d_avg_pct: float | None = Field(
        None, description="average daily volatility over the next 5 days, %"
    )
    vol_annualized_pct: float | None = None
    risk: list[RiskLevel] = Field(default_factory=list)
    generated_at_utc: str
    data_source: str
    disclaimer: str = (
        "Research backtest output from free daily data. Not investment advice and not a production risk system."
    )


class AllModelsResponse(BaseModel):
    asset: str
    forecasts: list[ForecastResponse]


class AssetInfo(BaseModel):
    asset: str
    name: str
    ticker: str
    asset_class: str
    last_close_date: str
    forecast_for: str


class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model: str
    label: str
    has_var: bool


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    forecasts_loaded: bool
    forecasts_file: str
    generated_at_utc: str | None = None
    data_source: str | None = None
    n_assets: int = 0
    days_since_last_close: int | None = Field(
        None, description="staleness check for the scheduled refresh"
    )
    version: str
