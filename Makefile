# One-command entry points. Override the config with e.g. `make all CONFIG=config/synthetic.yaml`.
PY      ?= python
CONFIG  ?= config/default.yaml
VOLTAIL  = $(PY) -m voltail.pipeline --config $(CONFIG)

.PHONY: help install data fit backtest report forecast all serve test test-fast lint format synthetic docker clean-results

help:  ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

install:  ## install pinned dependencies + package in editable mode
	$(PY) -m pip install -r requirements.lock
	$(PY) -m pip install -e . --no-deps

data:  ## download (or simulate) and clean daily OHLC -> parquet, logs dropped rows
	$(VOLTAIL) data

fit:  ## full-sample model fits (parameter tables, scratch-vs-arch check) + latest next-day forecasts
	$(VOLTAIL) fit

backtest:  ## walk-forward out-of-sample forecasts for every model/asset + statistical evaluation
	$(VOLTAIL) backtest

report:  ## build reports/RESULTS.md + figures and refresh AUTO blocks in README/docs
	$(VOLTAIL) report

forecast:  ## refresh only the next-day forecasts served by the API
	$(VOLTAIL) forecast

all: data backtest fit report  ## everything, end to end

serve:  ## run the FastAPI service on :8000
	$(PY) -m uvicorn voltail.api.app:app --host 0.0.0.0 --port 8000

test:  ## full pytest suite (includes slow end-to-end tests)
	$(PY) -m pytest

test-fast:  ## skip slow tests
	$(PY) -m pytest -m "not slow"

lint:  ## ruff + black --check
	ruff check src tests
	black --check src tests

format:  ## auto-format
	ruff check --fix src tests
	black src tests

synthetic:  ## full pipeline on simulated data (writes to */synthetic/, never touches README)
	$(MAKE) all CONFIG=config/synthetic.yaml

docker:  ## build and start the API container
	docker compose up --build

clean-results:  ## remove generated results (keeps cached data)
	rm -rf results/forecasts results/grids results/eval results/fit results/params.parquet results/diagnostics.json results/xgb_params.json reports/RESULTS.md reports/figures
