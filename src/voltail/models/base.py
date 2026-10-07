"""Common interface for model families used by the walk-forward engine.

Timing convention (the single most important thing in this project)
------------------------------------------------------------------
An *origin* is a trading day t at whose close a forecast is made. The forecast
targets day t+1 (next-day) or days t+1..t+5 (next-week). A family's
:meth:`ModelFamily.run_block` receives

* ``hist``       - rows 0..n-1 of the asset's data (everything up to the last origin
                   in the block; nothing later is ever passed in), and
* ``train_end``  - position in ``hist`` of the last row that may be used to
                   *estimate parameters*.

It must return forecasts for origins ``train_end, ..., n-1``. Parameters are
estimated on ``hist.iloc[:train_end+1]`` only; the forecast at origin o may use
``hist.iloc[:o+1]`` (filtering with fixed parameters). The engine then labels
the forecast for origin o with the date of row o+1. ``tests/test_no_lookahead.py``
checks this by perturbing future data and asserting earlier forecasts are unchanged.

Column naming in the output
---------------------------
``var1_<model>``          next-day variance forecast (percent^2)
``var5_<model>``          average daily variance over the next 5 days (percent^2)
``VaR_<a>_<model>``       1-day VaR at tail prob a (e.g. a010 = 1%), positive loss in %
``ES_<a>_<model>``        1-day Expected Shortfall, positive loss in %
plus a per-model tail-quantile grid (return units) used by the ES Monte Carlo test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from voltail.models.distributions import alpha_label


@dataclass
class BlockOutput:
    """Forecasts for one block of origins."""

    cols: dict[str, np.ndarray] = field(default_factory=dict)
    grids: dict[str, np.ndarray] = field(
        default_factory=dict
    )  # model -> (n_origins, K) return quantiles
    params: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def add_location_scale_risk(
        self,
        model: str,
        mu: np.ndarray,
        sigma: np.ndarray,
        alphas: list[float],
        q_fn,
        es_fn,
        grid_u: np.ndarray,
    ) -> None:
        """VaR/ES/tail grid for r = mu + sigma z with standardized quantile ``q_fn`` and ES ``es_fn``."""
        for a in alphas:
            lab = alpha_label(a)
            self.cols[f"VaR_{lab}_{model}"] = -(mu + sigma * q_fn(a))
            self.cols[f"ES_{lab}_{model}"] = -mu + sigma * es_fn(a)
        self.grids[model] = mu[:, None] + sigma[:, None] * np.asarray(q_fn(grid_u))[None, :]


class ModelFamily:
    """Base class. Subclasses set ``name``, ``refit_key`` and implement ``run_block``."""

    name: str = "base"
    refit_key: str | None = None  # key into cfg['walkforward']['refit_every']; None = never refit

    def __init__(self, cfg: dict[str, Any], asset: str, context: dict[str, Any] | None = None):
        self.cfg = cfg
        self.asset = asset
        self.context = context or {}
        self.alphas: list[float] = list(cfg["risk"]["alphas"])
        from voltail.models.distributions import tail_grid

        self.grid_u = tail_grid(cfg["models"]["tail_quantile_grid"], max(self.alphas))
        self.state: dict[str, Any] = {}

    def run_block(self, hist: pd.DataFrame, train_end: int) -> BlockOutput:  # pragma: no cover
        raise NotImplementedError
