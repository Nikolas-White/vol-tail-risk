"""Which family produces which model, and how to instantiate the families for an asset."""

from __future__ import annotations

from typing import Any

import pandas as pd

from voltail.models.base import ModelFamily
from voltail.models.baselines import BaselineFamily
from voltail.models.garch_arch import SPECS, TAIL_FILTERS, ArchFamily
from voltail.models.har import HARFamily
from voltail.models.ml import XGBFamily

MODEL_LABELS: dict[str, str] = {
    "roll21": "Rolling SD (21d)",
    "ewma": "EWMA / RiskMetrics (lambda=0.94)",
    "hs250": "Historical simulation (250d)",
    "garch_n": "GARCH(1,1)-normal",
    "garch_t": "GARCH(1,1)-t",
    "gjr_n": "GJR-GARCH-normal",
    "gjr_t": "GJR-GARCH-t",
    "gjr_skewt": "GJR-GARCH-skewed-t",
    "egarch_t": "EGARCH-t",
    "fhs_gjr": "Filtered HS (GJR-t filter)",
    "evt_gjr": "Conditional EVT/GPD (GJR-t filter)",
    "fhs_skewt": "Filtered HS (GJR-skew-t filter)",
    "evt_skewt": "Conditional EVT/GPD (GJR-skew-t filter)",
    "har": "HAR on Garman-Klass (log, OLS)",
    "xgb": "XGBoost on lagged features",
}

MODEL_FAMILY: dict[str, str] = {
    **{m: "baseline" for m in ["roll21", "ewma", "hs250"]},
    **{m: "garch" for m in [*SPECS, "fhs_gjr", "evt_gjr", "fhs_skewt", "evt_skewt"]},
    "har": "har",
    "xgb": "xgb",
}


def build_families(
    cfg: dict[str, Any], asset: str, driver: pd.DataFrame | None
) -> list[ModelFamily]:
    """Instantiate only the families needed for the configured vol/VaR model lists."""
    wanted = set(cfg["models"]["vol"]) | set(cfg["models"]["var"])
    needed = {MODEL_FAMILY[m] for m in wanted}
    fams: list[ModelFamily] = []
    if "baseline" in needed:
        fams.append(BaselineFamily(cfg, asset))
    if "garch" in needed:
        arch_models = [m for m in SPECS if m in wanted]
        for filt, tail_models in TAIL_FILTERS.items():
            if set(tail_models) & wanted and filt not in arch_models:
                arch_models.append(filt)
        fams.append(ArchFamily(cfg, asset, models=arch_models))
    if "har" in needed:
        fams.append(HARFamily(cfg, asset))
    if "xgb" in needed:
        fams.append(XGBFamily(cfg, asset, context={"driver": driver}))
    return fams
