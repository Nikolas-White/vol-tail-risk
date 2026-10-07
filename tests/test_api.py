"""API endpoint tests against a small forecasts file."""

from __future__ import annotations

import json
from datetime import date

import pytest
from fastapi.testclient import TestClient

from voltail.api import app as app_module

PAYLOAD = {
    "generated_at_utc": "2026-01-02T22:00:00+00:00",
    "run_name": "test",
    "data_source": "synthetic",
    "default_model": "gjr_t",
    "assets": {
        "SPX": {
            "name": "S&P 500 index",
            "ticker": "^GSPC",
            "asset_class": "equity",
            "last_close_date": str(date.today()),
            "forecast_for": "2026-01-05",
            "last_return_pct": -0.5,
            "models": {
                "gjr_t": {
                    "label": "GJR-GARCH-t",
                    "vol_1d": 1.1,
                    "vol_5d_avg": 1.0,
                    "vol_annualized": 17.5,
                    "var_990": 2.9,
                    "es_990": 3.6,
                    "var_975": 2.3,
                    "es_975": 2.9,
                },
                "roll21": {
                    "label": "Rolling SD (21d)",
                    "vol_1d": 0.9,
                    "vol_5d_avg": 0.9,
                    "vol_annualized": 14.3,
                },
            },
        }
    },
}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    f = tmp_path / "latest.json"
    f.write_text(json.dumps(PAYLOAD))
    monkeypatch.setattr(app_module, "store", app_module.ForecastStore(f))
    return TestClient(app_module.app), f


def test_health_ok(client):
    c, _ = client
    r = c.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["n_assets"] == 1 and body["days_since_last_close"] == 0


def test_forecast_default_model(client):
    c, _ = client
    r = c.get("/forecast/SPX")
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "gjr_t"
    assert body["vol_1d_pct"] == 1.1
    assert {x["confidence"] for x in body["risk"]} == {0.99, 0.975}
    assert all(x["es_pct"] >= x["var_pct"] for x in body["risk"])
    assert body["data_source"] == "synthetic"


def test_forecast_case_insensitive_and_model_param(client):
    c, _ = client
    r = c.get("/forecast/spx", params={"model": "roll21"})
    assert r.status_code == 200
    assert r.json()["risk"] == []  # volatility-only model


def test_unknown_asset_404(client):
    c, _ = client
    assert c.get("/forecast/NOPE").status_code == 404


def test_unknown_model_404(client):
    c, _ = client
    assert c.get("/forecast/SPX", params={"model": "lstm"}).status_code == 404


@pytest.mark.parametrize("bad", ["S", "SPX%20DROP", "a" * 20])
def test_invalid_asset_rejected_422(client, bad):
    c, _ = client
    assert c.get(f"/forecast/{bad}").status_code in (404, 422)
    if bad != "SPX%20DROP":
        assert c.get(f"/forecast/{bad}").status_code == 422


def test_invalid_model_pattern_422(client):
    c, _ = client
    assert c.get("/forecast/SPX", params={"model": "DROP TABLE"}).status_code == 422


def test_all_models(client):
    c, _ = client
    r = c.get("/forecast/SPX/all")
    assert r.status_code == 200
    assert {f["model"] for f in r.json()["forecasts"]} == {"gjr_t", "roll21"}


def test_assets_and_models(client):
    c, _ = client
    assert c.get("/assets").json()[0]["asset"] == "SPX"
    models = {m["model"]: m["has_var"] for m in c.get("/models").json()}
    assert models == {"gjr_t": True, "roll21": False}


def test_missing_file_returns_503_and_health_degrades(client):
    c, f = client
    f.unlink()
    assert c.get("/forecast/SPX").status_code == 503
    assert c.get("/health").json()["status"] == "no_forecasts"


def test_file_reloaded_when_changed(client):
    c, f = client
    import os
    import time

    new = json.loads(json.dumps(PAYLOAD))
    new["assets"]["SPX"]["models"]["gjr_t"]["vol_1d"] = 2.2
    f.write_text(json.dumps(new))
    os.utime(f, (time.time() + 5, time.time() + 5))
    assert c.get("/forecast/SPX").json()["vol_1d_pct"] == 2.2
