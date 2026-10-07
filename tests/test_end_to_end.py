"""End-to-end: run the whole pipeline on a small synthetic universe (config/smoke.yaml).

Checks that every stage runs, that outputs have the expected structure, that the
data-generating parameters are recovered by the full-sample fit, and that a
synthetic run never writes into README/docs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from voltail.config import PROJECT_ROOT, load_config
from voltail.data.synthetic import CLASS_PARAMS
from voltail.pipeline import main
from voltail.reporting.inject import inject_blocks

pytestmark = pytest.mark.slow


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else ""


@pytest.fixture(scope="module")
def smoke_run():
    protected = [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "docs" / "MODEL_VALIDATION_REPORT.md",
        PROJECT_ROOT / "RESUME_BULLETS.md",
    ]
    before = {p: _sha(p) for p in protected}
    for stage in ["data", "backtest", "fit", "report"]:
        main(["--config", "config/smoke.yaml", stage])
    cfg = load_config("config/smoke.yaml")
    return cfg, before, protected


def test_outputs_exist(smoke_run):
    cfg, _, _ = smoke_run
    res = Path(cfg["paths"]["results_dir"])
    for asset in cfg["data"]["assets"]:
        assert (res / "forecasts" / f"{asset}.parquet").exists()
    for t in ["vol_metrics", "var_backtests", "mcs_vol", "mz", "ratings"]:
        assert (res / "eval" / f"{t}.parquet").exists()
    report = Path(cfg["paths"]["reports_dir"]) / "RESULTS.md"
    text = report.read_text()
    assert "SYNTHETIC DATA" in text
    assert "## 4. VaR and ES backtests" in text


def test_synthetic_run_does_not_touch_docs(smoke_run):
    _, before, protected = smoke_run
    for p in protected:
        assert _sha(p) == before[p], f"{p.name} was modified by a synthetic run"


def test_latest_forecasts_json(smoke_run):
    cfg, _, _ = smoke_run
    payload = json.loads((PROJECT_ROOT / cfg["service"]["forecasts_file"]).read_text())
    spx = payload["assets"]["SPX"]["models"]["gjr_t"]
    assert spx["vol_1d"] > 0 and spx["var_990"] > spx["var_975"] > 0
    assert spx["es_990"] > spx["var_990"]  # ES is always beyond VaR


def test_full_sample_fit_recovers_generating_parameters(smoke_run):
    cfg, _, _ = smoke_run
    p = pd.read_parquet(Path(cfg["paths"]["results_dir"]) / "fit" / "fullsample_params.parquet")
    g = p[(p["asset"] == "SPX") & (p["model"] == "gjr_t")].set_index("param")
    true = CLASS_PARAMS["equity"]
    # ~5 years of data: loose but meaningful bands (several standard errors)
    assert abs(g.loc["persistence", "value"] - true.persistence) < 0.04
    assert abs(g.loc["gamma[1]", "value"] - true.gamma) < 4 * g.loc["gamma[1]", "se"]
    assert abs(g.loc["beta[1]", "value"] - true.beta) < 4 * g.loc["beta[1]", "se"]


def test_scratch_garch_reconciles_with_arch_on_every_asset(smoke_run):
    cfg, _, _ = smoke_run
    r = pd.read_parquet(Path(cfg["paths"]["results_dir"]) / "fit" / "garch_reconciliation.parquet")
    assert (r.loc[r["param"].isin(["alpha", "beta"]), "abs_diff"] < 5e-3).all()
    assert (r.loc[r["param"] == "loglik", "abs_diff"] < 0.1).all()


def test_true_model_family_beats_naive_baselines(smoke_run):
    """On GJR-GARCH data, GJR-t should have lower QLIKE than a 21-day rolling SD."""
    cfg, _, _ = smoke_run
    vm = pd.read_parquet(Path(cfg["paths"]["results_dir"]) / "eval" / "vol_metrics.parquet")
    full = vm[(vm["regime"] == "Full") & (vm["horizon"] == 1) & (vm["proxy"] == "gk")].set_index(
        ["asset", "model"]
    )
    for asset in cfg["data"]["assets"]:
        assert full.loc[(asset, "gjr_t"), "qlike"] < full.loc[(asset, "roll21"), "qlike"]


def test_inject_blocks_only_touches_markers():
    text = (
        "intro\n<!-- AUTO:x -->\nold\n<!-- /AUTO:x -->\noutro <!-- AUTO:y -->keep<!-- /AUTO:y -->"
    )
    new, replaced = inject_blocks(text, {"x": "NEW"})
    assert replaced == ["x"]
    assert (
        "NEW" in new
        and "old" not in new
        and "keep" in new
        and new.startswith("intro")
        and "outro" in new
    )


def test_production_reuses_frozen_xgb_hyperparameters(smoke_run):
    cfg, _, _ = smoke_run
    frozen = json.loads((Path(cfg["paths"]["results_dir"]) / "xgb_params.json").read_text())
    assert set(frozen) == set(cfg["data"]["assets"])
    assert {"params_h1", "params_h5"} <= set(frozen["SPX"])
