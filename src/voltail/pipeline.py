"""Command-line entry point: ``python -m voltail.pipeline --config <yaml> <stage>``.

Stages
------
data      download (or simulate) + clean -> data/processed/*.parquet
backtest  walk-forward out-of-sample forecasts + statistical evaluation
fit       full-sample fits (parameter tables, scratch-vs-arch check) + next-day forecasts
forecast  next-day forecasts only (used by the scheduled GitHub Action)
report    figures, reports/RESULTS.md, and AUTO blocks in README/docs
"""

from __future__ import annotations

import argparse
import time

from voltail.config import ensure_dirs, load_config, set_seeds
from voltail.logging_utils import get_logger

log = get_logger("voltail")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="voltail", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--config", default=None, help="override YAML deep-merged onto config/default.yaml"
    )
    ap.add_argument("stage", choices=["data", "backtest", "fit", "forecast", "report"])
    ap.add_argument("--assets", nargs="*", default=None, help="restrict to these assets")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.assets:
        cfg["data"]["assets"] = {a: s for a, s in cfg["data"]["assets"].items() if a in args.assets}
    set_seeds(cfg["seed"])
    ensure_dirs(cfg)
    t0 = time.time()
    log.info(
        "stage=%s run=%s source=%s assets=%s",
        args.stage,
        cfg["run_name"],
        cfg["data"]["source"],
        list(cfg["data"]["assets"]),
    )

    if args.stage == "data":
        from voltail.data.store import run_data_stage

        run_data_stage(cfg)
    elif args.stage == "backtest":
        from voltail.backtest.runner import run_backtest_stage

        run_backtest_stage(cfg)
    elif args.stage == "fit":
        from voltail.backtest.production import run_fit_stage

        run_fit_stage(cfg)
    elif args.stage == "forecast":
        from voltail.backtest.production import run_forecast_stage

        run_forecast_stage(cfg)
    elif args.stage == "report":
        from voltail.reporting.report import run_report_stage

        run_report_stage(cfg)
    log.info("stage=%s finished in %.1fs", args.stage, time.time() - t0)


if __name__ == "__main__":
    main()
