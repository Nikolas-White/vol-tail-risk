"""Synthetic OHLC data from *known* GJR-GARCH-t processes.

Used for (a) unit/end-to-end tests where the true parameters are known and
(b) a full dry run of the pipeline when real data cannot be downloaded.
Synthetic results are always written to separate directories and stamped
"SYNTHETIC" in every report; they are never presented as market results.

Construction for each day t (returns in percent):

* conditional variance  h_t = w + (a + g 1[e_{t-1}<0]) e_{t-1}^2 + b h_{t-1}
* total return          r_t = mu + e_t,  e_t = sqrt(h_t) z_t,  z_t ~ unit-variance Student-t
* an overnight share ``f`` of the shock is the gap between the previous close and
  today's open; the rest is spread over an intraday Brownian bridge with ``n_steps``
  points pinned to the close, which gives realistic high/low ranges.

Shocks are cross-correlated through a one-factor Gaussian structure before the
Student-t scaling, so cross-asset features have something real to find.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar


@dataclass(frozen=True)
class GJRParams:
    """Parameters of a GJR-GARCH(1,1)-t process in percent units."""

    mu: float
    omega: float
    alpha: float
    gamma: float
    beta: float
    nu: float
    overnight_share: float
    start_price: float

    @property
    def persistence(self) -> float:
        return self.alpha + 0.5 * self.gamma + self.beta

    @property
    def uncond_var(self) -> float:
        return self.omega / (1.0 - self.persistence)


CLASS_PARAMS: dict[str, GJRParams] = {
    "equity": GJRParams(0.04, 0.020, 0.010, 0.140, 0.890, 6.0, 0.20, 1200.0),
    "fx": GJRParams(0.00, 0.004, 0.030, 0.000, 0.960, 8.0, 0.02, 1.25),
    "commodity": GJRParams(0.02, 0.020, 0.040, 0.000, 0.945, 6.0, 0.10, 90.0),
    "rates": GJRParams(0.01, 0.010, 0.040, 0.000, 0.950, 8.0, 0.10, 90.0),
    "crypto": GJRParams(0.10, 0.400, 0.100, 0.000, 0.870, 3.5, 0.00, 400.0),
}
FACTOR_LOADING = {"equity": 0.75, "fx": 0.25, "commodity": 0.2, "rates": -0.3, "crypto": 0.3}


def trading_calendar(start: str, end: str, asset_class: str) -> pd.DatetimeIndex:
    """Business days minus US federal holidays (all days for crypto)."""
    if asset_class == "crypto":
        return pd.date_range(start, end, freq="D", name="date")
    bdays = pd.bdate_range(start, end, name="date")
    hol = USFederalHolidayCalendar().holidays(start, end)
    return bdays.difference(hol).rename("date")


def simulate_gjr_t(
    n: int, p: GJRParams, rng: np.random.Generator, normals: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate ``n`` returns and conditional variances from a GJR-GARCH-t process.

    ``normals`` (optional, shape (n,)) supplies correlated standard normals; they are
    converted to unit-variance Student-t draws by dividing by sqrt(chi2_nu / (nu-2)).
    """
    z_n = rng.standard_normal(n) if normals is None else normals
    chi = rng.chisquare(p.nu, size=n)
    z = z_n / np.sqrt(chi / (p.nu - 2.0))
    h = np.empty(n)
    e = np.empty(n)
    h_prev = p.uncond_var
    e_prev = 0.0
    for t in range(n):
        h_t = p.omega + (p.alpha + p.gamma * (e_prev < 0)) * e_prev**2 + p.beta * h_prev
        e_t = np.sqrt(h_t) * z[t]
        h[t], e[t] = h_t, e_t
        h_prev, e_prev = h_t, e_t
    return p.mu + e, h


def simulate_universe(cfg: dict[str, Any]) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Simulate raw OHLC frames for every configured asset.

    Returns ``(frames, truth)`` where ``truth`` holds the generating parameters and
    the list of injected data defects (used to test the cleaning step).
    """
    rng = np.random.default_rng(cfg["seed"])
    start, end = cfg["data"]["start"], cfg["data"]["end"] or "2026-09-30"
    full_days = pd.date_range(start, end, freq="D")
    factor = rng.standard_normal(len(full_days))
    frames: dict[str, pd.DataFrame] = {}
    true_var: dict[str, pd.Series] = {}
    truth: dict[str, Any] = {
        "generator": "GJR-GARCH(1,1)-t with intraday Brownian bridge",
        "assets": {},
    }
    for asset, spec in cfg["data"]["assets"].items():
        cls = spec["asset_class"]
        p = CLASS_PARAMS[cls]
        a_start = (
            "2014-09-17"
            if cls == "crypto" and pd.Timestamp(start) < pd.Timestamp("2014-09-17")
            else start
        )
        cal = trading_calendar(a_start, end, cls)
        load = FACTOR_LOADING[cls]
        f_today = pd.Series(factor, index=full_days).reindex(cal).to_numpy()
        normals = load * f_today + np.sqrt(1 - load**2) * rng.standard_normal(len(cal))
        # simulate actual vol-driven returns and attach realistic intraday ranges
        ret, h = simulate_gjr_t(len(cal), p, rng, normals)
        frames[asset] = ohlc_from_returns(ret, h, p, rng, cal)
        true_var[asset] = pd.Series(h, index=cal, name="true_var")
        truth["assets"][asset] = {"asset_class": cls, **asdict(p), "persistence": p.persistence}

    defects: dict[str, list[str]] = {}
    for asset, df in frames.items():
        if cfg["data"]["assets"][asset]["asset_class"] == "fx" and len(df) > 1000:
            # inject defects the cleaner must catch: zero-range rows and one spike-and-revert bad tick
            idx = rng.choice(np.arange(300, len(df) - 300), size=12, replace=False)
            zero_rng = df.index[idx[:10]]
            df.loc[zero_rng, ["open", "high", "low"]] = df.loc[zero_rng, "close"].to_numpy()[
                :, None
            ]
            spike_day = df.index[idx[10]]
            df.loc[spike_day, ["open", "high", "low", "close"]] = df.loc[spike_day, "close"] * 1.25
            defects[asset] = {
                "zero_range": [str(d.date()) for d in zero_rng],
                "spike": [str(spike_day.date())],
            }
    truth["injected_defects"] = defects
    truth["true_variance"] = true_var  # conditional variance h_t of the return on day t (percent^2)
    return frames, truth


def ohlc_from_returns(
    ret_pct: np.ndarray,
    h: np.ndarray,
    p: GJRParams,
    rng: np.random.Generator,
    index: pd.DatetimeIndex,
    n_steps: int = 78,
) -> pd.DataFrame:
    """Build OHLC bars consistent with the simulated close-to-close returns.

    A fixed share ``f`` of each day's log move happens overnight (close -> open);
    the rest is the open -> close move. High/low come from a Brownian bridge with
    per-step sd sqrt((1-f) h_t / n_steps), pinned at the open -> close move, so the
    range reflects the *conditional* variance (like real intraday trading) and
    range proxies are biased low versus r^2 by roughly the overnight share.
    """
    n = len(ret_pct)
    r = ret_pct / 100.0
    overnight = p.overnight_share * r
    intraday = r - overnight
    close = p.start_price * np.exp(np.cumsum(r))
    prev_close = np.concatenate([[p.start_price], close[:-1]])
    open_ = prev_close * np.exp(overnight)
    sd_step = np.sqrt((1.0 - p.overnight_share) * h) / 100.0 / np.sqrt(n_steps)
    walk = np.cumsum(rng.standard_normal((n, n_steps)) * sd_step[:, None], axis=1)
    k = np.arange(1, n_steps + 1)[None, :] / n_steps
    bridge = walk - k * (walk[:, -1:] - intraday[:, None])
    high = open_ * np.exp(np.maximum(bridge.max(axis=1), np.maximum(intraday, 0.0)))
    low = open_ * np.exp(np.minimum(bridge.min(axis=1), np.minimum(intraday, 0.0)))
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 0.0},
        index=index.rename("date"),
    )
