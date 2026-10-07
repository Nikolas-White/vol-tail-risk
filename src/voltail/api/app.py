"""FastAPI service serving the latest next-day volatility / VaR / ES forecasts.

The service is deliberately thin: it never fits models at request time. It reads
``results/latest_forecasts.json`` (written by ``make forecast`` and refreshed by the
scheduled GitHub Action) and reloads it automatically when the file changes.

Endpoints
---------
GET /health                       liveness + staleness information
GET /assets                       assets with forecasts
GET /models                       available models
GET /forecast/{asset}?model=...   one model's forecast (default model from config)
GET /forecast/{asset}/all         every model's forecast for an asset

Run:  ``make serve``  or  ``uvicorn voltail.api.app:app``.
Override the file with the ``VOLTAIL_FORECASTS_FILE`` environment variable.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import date
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi import Path as PathParam

from voltail import __version__
from voltail.api.schemas import (
    ASSET_PATTERN,
    MODEL_PATTERN,
    AllModelsResponse,
    AssetInfo,
    ForecastResponse,
    HealthResponse,
    ModelInfo,
    RiskLevel,
)
from voltail.config import PROJECT_ROOT, load_config


def _forecasts_path() -> Path:
    env = os.environ.get("VOLTAIL_FORECASTS_FILE")
    p = Path(env) if env else Path(load_config()["service"]["forecasts_file"])
    return p if p.is_absolute() else PROJECT_ROOT / p


class ForecastStore:
    """Caches the JSON payload; reloads when the file's modification time changes."""

    def __init__(self, path: Path):
        self.path = path
        self._mtime: float | None = None
        self._data: dict[str, Any] | None = None
        self._lock = threading.Lock()

    def get(self) -> dict[str, Any] | None:
        with self._lock:
            if not self.path.exists():
                self._data, self._mtime = None, None
                return None
            mtime = self.path.stat().st_mtime
            if mtime != self._mtime:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
                self._mtime = mtime
            return self._data


app = FastAPI(
    title="voltail forecast API",
    version=__version__,
    description="Next-day volatility, Value-at-Risk and Expected Shortfall forecasts from backtested models.",
)
store = ForecastStore(_forecasts_path())


def _payload() -> dict[str, Any]:
    data = store.get()
    if data is None:
        raise HTTPException(
            status_code=503,
            detail=f"No forecasts available yet ({store.path.name} missing). Run `make forecast`.",
        )
    return data


def _asset_entry(data: dict[str, Any], asset: str) -> dict[str, Any]:
    key = asset.upper()
    if key not in data["assets"]:
        raise HTTPException(
            status_code=404, detail=f"Unknown asset '{asset}'. Available: {sorted(data['assets'])}"
        )
    return data["assets"][key]


def _to_response(
    data: dict[str, Any], asset: str, entry: dict[str, Any], model: str
) -> ForecastResponse:
    if model not in entry["models"]:
        raise HTTPException(
            status_code=404, detail=f"Unknown model '{model}'. Available: {sorted(entry['models'])}"
        )
    m = entry["models"][model]
    risk = []
    for level in (990, 975):
        if f"var_{level}" in m:
            risk.append(
                RiskLevel(
                    confidence=level / 1000, var_pct=m[f"var_{level}"], es_pct=m[f"es_{level}"]
                )
            )
    return ForecastResponse(
        asset=asset.upper(),
        asset_name=entry["name"],
        model=model,
        model_label=m.get("label", model),
        last_close_date=entry["last_close_date"],
        forecast_for=entry["forecast_for"],
        vol_1d_pct=m.get("vol_1d"),
        vol_5d_avg_pct=m.get("vol_5d_avg"),
        vol_annualized_pct=m.get("vol_annualized"),
        risk=risk,
        generated_at_utc=data["generated_at_utc"],
        data_source=data["data_source"],
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    data = store.get()
    if data is None:
        return HealthResponse(
            status="no_forecasts",
            forecasts_loaded=False,
            forecasts_file=str(store.path.name),
            version=__version__,
        )
    last = max(date.fromisoformat(a["last_close_date"]) for a in data["assets"].values())
    return HealthResponse(
        status="ok",
        forecasts_loaded=True,
        forecasts_file=str(store.path.name),
        generated_at_utc=data["generated_at_utc"],
        data_source=data["data_source"],
        n_assets=len(data["assets"]),
        days_since_last_close=(date.today() - last).days,
        version=__version__,
    )


@app.get("/assets", response_model=list[AssetInfo])
def assets() -> list[AssetInfo]:
    data = _payload()
    return [
        AssetInfo(
            asset=k,
            name=v["name"],
            ticker=v["ticker"],
            asset_class=v["asset_class"],
            last_close_date=v["last_close_date"],
            forecast_for=v["forecast_for"],
        )
        for k, v in data["assets"].items()
    ]


@app.get("/models", response_model=list[ModelInfo])
def models() -> list[ModelInfo]:
    data = _payload()
    first = next(iter(data["assets"].values()))
    return [
        ModelInfo(model=k, label=v.get("label", k), has_var="var_990" in v)
        for k, v in first["models"].items()
    ]


@app.get("/forecast/{asset}", response_model=ForecastResponse)
def forecast(
    asset: Annotated[str, PathParam(pattern=ASSET_PATTERN, description="asset code, e.g. SPX")],
    model: Annotated[
        str | None, Query(pattern=MODEL_PATTERN, description="model id, e.g. gjr_t")
    ] = None,
) -> ForecastResponse:
    data = _payload()
    entry = _asset_entry(data, asset)
    return _to_response(data, asset, entry, model or data["default_model"])


@app.get("/forecast/{asset}/all", response_model=AllModelsResponse)
def forecast_all(asset: Annotated[str, PathParam(pattern=ASSET_PATTERN)]) -> AllModelsResponse:
    data = _payload()
    entry = _asset_entry(data, asset)
    return AllModelsResponse(
        asset=asset.upper(),
        forecasts=[_to_response(data, asset, entry, m) for m in entry["models"]],
    )
