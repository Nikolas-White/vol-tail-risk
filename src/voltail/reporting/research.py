"""Pre-registered research question: is the equity VaR failure a skewness problem?

Background (from the first real-data run, before this analysis existed): GJR-GARCH with
symmetric Student-t innovations forecast volatility well but exceeded its 99% VaR on
~1.6-1.8% of days for the S&P 500 and TSX (Basel red zone). Filtered HS, which uses the
same GJR volatility but the *empirical* residual distribution, was close to 1%. So the
scale was right and the *shape* of the assumed distribution was wrong. Equity returns
are negatively skewed (crashes are bigger than rallies), which a symmetric t cannot fit.

Hypothesis H1 (written down in docs/METHODOLOGY.md section 9 before the skewed-t
model was run on real data): replacing the symmetric t with Hansen's skewed t fixes it.

Decision rule, applied mechanically to each equity index (SPX, TSX):

C1  full-sample skew parameter lambda < 0 with |lambda / SE| > 2
C2  at 99%, the skewed-t model passes Kupiec (p >= 0.05) and is not in the red zone
C3  at 99%, the skewed-t model has lower FZ0 loss than the symmetric-t model, with a
    one-sided Diebold-Mariano p-value < 0.05

Verdict: SUPPORTED if C1-C3 hold for both indices; PARTIALLY SUPPORTED if C1 holds for
both and C2 or C3 holds for at least one; NOT SUPPORTED otherwise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from voltail.evaluation.forecast_tests import diebold_mariano
from voltail.evaluation.losses import fz0
from voltail.evaluation.var_tests import traffic_light, var_backtest
from voltail.models.distributions import alpha_label

EQUITY = ["SPX", "TSX"]
BASE, CHALLENGER = "gjr_t", "gjr_skewt"


def skew_comparison(cfg: dict[str, Any], forecasts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per asset and level: exceedances, Kupiec, zone and FZ0 DM test for skew-t vs t."""
    rows = []
    for asset, fc in forecasts.items():
        for a in cfg["risk"]["alphas"]:
            lab = alpha_label(a)
            cols = [f"{k}_{lab}_{m}" for k in ("VaR", "ES") for m in (BASE, CHALLENGER)]
            if not set(cols) <= set(fc.columns):
                continue
            s = fc[cols + ["ret"]].dropna()
            r = s["ret"].to_numpy()
            row: dict[str, Any] = {"asset": asset, "level": 1 - a, "n": len(s)}
            losses = {}
            for m in (BASE, CHALLENGER):
                bt = var_backtest(r, s[f"VaR_{lab}_{m}"].to_numpy(), a)
                row[f"rate_{m}"] = bt.rate
                row[f"p_kupiec_{m}"] = bt.p_pof
                row[f"zone_{m}"] = traffic_light(bt.exceedances, bt.n, a)
                losses[m] = fz0(r, s[f"VaR_{lab}_{m}"].to_numpy(), s[f"ES_{lab}_{m}"].to_numpy(), a)
            dm = diebold_mariano(losses[CHALLENGER], losses[BASE])
            row["fz0_diff"] = dm.mean_diff  # negative = skew-t better
            row["dm_stat"] = dm.stat
            row["dm_p_one_sided"] = dm.p_value / 2 if dm.stat < 0 else 1 - dm.p_value / 2
            rows.append(row)
    return pd.DataFrame(rows)


def lambda_table(cfg: dict[str, Any]) -> pd.DataFrame:
    """Full-sample skew estimate (with SE) and its range across monthly refits."""
    res = Path(cfg["paths"]["results_dir"])
    full = pd.read_parquet(res / "fit" / "fullsample_params.parquet")
    full = full[(full["model"] == CHALLENGER) & (full["param"] == "lambda")].set_index("asset")
    refits = pd.read_parquet(res / "params.parquet")
    refits = refits[(refits["model"] == CHALLENGER) & (refits["param"] == "lambda")]
    g = refits.groupby("asset")["value"]
    out = pd.DataFrame(
        {
            "lambda_full_sample": full["value"],
            "se": full["se"],
            "t_stat": full["value"] / full["se"],
            "lambda_refit_min": g.min(),
            "lambda_refit_median": g.median(),
            "lambda_refit_max": g.max(),
        }
    )
    return out.reindex([a for a in cfg["data"]["assets"] if a in out.index])


def verdict(cmp: pd.DataFrame, lam: pd.DataFrame) -> tuple[str, list[str]]:
    """Apply the pre-registered decision rule. Returns (verdict, per-criterion explanations)."""
    lines, c1_all, c2_any, c3_any, all_three = [], True, False, False, True
    for asset in EQUITY:
        if asset not in lam.index:
            return "NOT EVALUATED (equity data missing)", []
        r = cmp[(cmp["asset"] == asset) & np.isclose(cmp["level"], 0.99)].iloc[0]
        c1 = bool(lam.loc[asset, "lambda_full_sample"] < 0 and abs(lam.loc[asset, "t_stat"]) > 2)
        c2 = bool(r[f"p_kupiec_{CHALLENGER}"] >= 0.05 and r[f"zone_{CHALLENGER}"] != "red")
        c3 = bool(r["fz0_diff"] < 0 and r["dm_p_one_sided"] < 0.05)
        c1_all &= c1
        c2_any |= c2
        c3_any |= c3
        all_three &= c1 and c2 and c3
        lines.append(
            f"{asset}: C1 {'met' if c1 else 'NOT met'} (lambda = {lam.loc[asset, 'lambda_full_sample']:.3f}, "
            f"t = {lam.loc[asset, 't_stat']:.1f}); "
            f"C2 {'met' if c2 else 'NOT met'} (99% exceedances {100 * r[f'rate_{BASE}']:.2f}% -> {100 * r[f'rate_{CHALLENGER}']:.2f}%, "
            f"zone {r[f'zone_{BASE}']} -> {r[f'zone_{CHALLENGER}']}, Kupiec p {r[f'p_kupiec_{CHALLENGER}']:.3f}); "
            f"C3 {'met' if c3 else 'NOT met'} (FZ0 difference {r['fz0_diff']:+.4f}, one-sided DM p {r['dm_p_one_sided']:.3f})."
        )
    if all_three:
        v = "SUPPORTED"
    elif c1_all and (c2_any or c3_any):
        v = "PARTIALLY SUPPORTED"
    else:
        v = "NOT SUPPORTED"
    return v, lines


def research_markdown(cfg: dict[str, Any], forecasts: dict[str, pd.DataFrame], lab_fn) -> str:
    cmp = skew_comparison(cfg, forecasts)
    if cmp.empty:
        return "_Skewed-t model not in this run._"
    lam = lambda_table(cfg)
    v, lines = verdict(cmp, lam)
    show = cmp.copy()
    for m in (BASE, CHALLENGER):
        show[f"rate_{m}"] = (100 * show[f"rate_{m}"]).map(lambda x: f"{x:.2f}%")
    show["level"] = show["level"].map(lambda x: f"{x:.1%}")
    show = show.rename(
        columns={
            f"rate_{BASE}": "exceed t",
            f"rate_{CHALLENGER}": "exceed skew-t",
            f"p_kupiec_{BASE}": "Kupiec p (t)",
            f"p_kupiec_{CHALLENGER}": "Kupiec p (skew-t)",
            f"zone_{BASE}": "zone t",
            f"zone_{CHALLENGER}": "zone skew-t",
            "fz0_diff": "FZ0 skew-t minus t",
            "dm_p_one_sided": "DM p (skew-t better)",
        }
    ).drop(columns=["dm_stat"])
    return "\n\n".join(
        [
            f"**Question.** {lab_fn(BASE)} forecast equity volatility well but breached its 99% VaR too often "
            "(red zone for the S&P 500 and TSX in the first run), while filtered HS with the same volatility did not. "
            "Is the failure caused by ignoring negative skewness? The hypothesis and the decision rule were written "
            "down before the skewed-t model was run (METHODOLOGY section 9, `src/voltail/reporting/research.py`).",
            f"**Verdict (mechanical, pre-registered rule): {v}.**",
            "\n".join(f"- {x}" for x in lines),
            "Estimated skewness (lambda < 0 = heavier left tail):",
            lam.reset_index()
            .rename(columns={"index": "asset"})
            .to_markdown(index=False, floatfmt=".3f"),
            f"{lab_fn(BASE)} vs {lab_fn(CHALLENGER)}, every asset and level:",
            show.to_markdown(index=False, floatfmt=".3f"),
        ]
    )


# ----------------------------------------------------------------------------- H2
H2_PAIRS = [("fhs_gjr", "fhs_skewt"), ("evt_gjr", "evt_skewt")]


def h2_comparison(
    cfg: dict[str, Any], forecasts: dict[str, pd.DataFrame], var_backtests: pd.DataFrame
) -> pd.DataFrame:
    """Per pair, asset and level: pass/fail under the rating rule and an FZ0 DM test."""
    from voltail.evaluation.ratings import cell_pass

    full = var_backtests[var_backtests["regime"] == "Full"].set_index(["model", "asset", "alpha"])
    rows = []
    for base, new in H2_PAIRS:
        for asset, fc in forecasts.items():
            for a in cfg["risk"]["alphas"]:
                lab = alpha_label(a)
                cols = [f"{k}_{lab}_{m}" for k in ("VaR", "ES") for m in (base, new)]
                if not set(cols) <= set(fc.columns) or (base, asset, a) not in full.index:
                    continue
                s = fc[cols + ["ret"]].dropna()
                r = s["ret"].to_numpy()
                lb = fz0(r, s[f"VaR_{lab}_{base}"].to_numpy(), s[f"ES_{lab}_{base}"].to_numpy(), a)
                ln = fz0(r, s[f"VaR_{lab}_{new}"].to_numpy(), s[f"ES_{lab}_{new}"].to_numpy(), a)
                dm = diebold_mariano(ln, lb)
                rows.append(
                    {
                        "pair": f"{base} -> {new}",
                        "asset": asset,
                        "level": 1 - a,
                        "rate_base": full.loc[(base, asset, a), "rate"],
                        "rate_new": full.loc[(new, asset, a), "rate"],
                        "pass_base": cell_pass(full.loc[(base, asset, a)]),
                        "pass_new": cell_pass(full.loc[(new, asset, a)]),
                        "fz0_diff": dm.mean_diff,
                        "dm_p_new_better": dm.p_value / 2 if dm.stat < 0 else 1 - dm.p_value / 2,
                        "dm_p_new_worse": dm.p_value / 2 if dm.stat > 0 else 1 - dm.p_value / 2,
                    }
                )
    return pd.DataFrame(rows)


def verdict_h2(cmp: pd.DataFrame) -> tuple[str, list[str]]:
    """Apply the pre-registered H2 rule (METHODOLOGY section 10)."""
    lines, e1_all, e2_all, any_hit, any_worse = [], True, True, False, False
    for pair, g in cmp.groupby("pair", sort=False):
        pb, pn = int(g["pass_base"].sum()), int(g["pass_new"].sum())
        better = int((g["dm_p_new_better"] < 0.05).sum())
        worse = int((g["dm_p_new_worse"] < 0.05).sum())
        e1, e2 = pn > pb, better > worse
        e1_all &= e1
        e2_all &= e2
        any_hit |= e1 or e2
        any_worse |= pn < pb
        lines.append(
            f"{pair}: E1 {'met' if e1 else 'NOT met'} (cells passing all tests {pb} -> {pn} of {len(g)}); "
            f"E2 {'met' if e2 else 'NOT met'} (FZ0 significantly better in {better} cells, significantly worse in {worse})."
        )
    if e1_all and e2_all:
        v = "SUPPORTED"
    elif any_hit and not any_worse:
        v = "PARTIALLY SUPPORTED"
    else:
        v = "NOT SUPPORTED"
    return v, lines


def research_h2_markdown(
    cfg: dict[str, Any], forecasts: dict[str, pd.DataFrame], var_backtests: pd.DataFrame, lab_fn
) -> str:
    cmp = h2_comparison(cfg, forecasts, var_backtests)
    if cmp.empty:
        return "_Skew-filtered tail models not in this run._"
    v, lines = verdict_h2(cmp)
    show = cmp.copy()
    show["level"] = show["level"].map(lambda x: f"{x:.1%}")
    for c in ("rate_base", "rate_new"):
        show[c] = (100 * show[c]).map(lambda x: f"{x:.2f}%")
    show = show.drop(columns=["dm_p_new_worse"]).rename(
        columns={
            "rate_base": "exceed (t filter)",
            "rate_new": "exceed (skew-t filter)",
            "pass_base": "passes (t filter)",
            "pass_new": "passes (skew-t filter)",
            "fz0_diff": "FZ0 new minus old",
            "dm_p_new_better": "DM p (skew-t filter better)",
        }
    )
    return "\n\n".join(
        [
            "**Question.** Does filtering the semi-parametric tail models (filtered HS, conditional EVT) with the "
            "skewed-t GJR model instead of the symmetric-t one improve them? Pre-registered in METHODOLOGY section 10, "
            "with the stated prior that it would change little (these models already take the tail shape from the data).",
            f"**Verdict (mechanical, pre-registered rule): {v}.**",
            "\n".join(f"- {x}" for x in lines),
            show.to_markdown(index=False, floatfmt=".3f"),
        ]
    )


# ----------------------------------------------------------------------------- placebo
def placebo_markdown(cfg: dict[str, Any]) -> tuple[str, str] | None:
    """Apply both decision rules to the synthetic run, where H1 and H2 are false by construction.

    The synthetic data uses *symmetric* Student-t innovations, so neither skewness nor a
    skewed-t filter can genuinely help there. Whatever verdict the rules return on it is
    what chance alone produces, and the real-data verdicts should be read against it.
    Returns (H1 placebo text, H2 placebo text), or None if no synthetic run exists.
    """
    if cfg["data"]["source"] == "synthetic":
        return None
    from voltail.backtest.runner import load_forecasts
    from voltail.config import PROJECT_ROOT, load_config

    scfg = load_config(str(PROJECT_ROOT / "config" / "synthetic.yaml"))
    res = Path(scfg["paths"]["results_dir"])
    if not (res / "fit" / "fullsample_params.parquet").exists() or not (res / "eval").exists():
        return None
    fc = load_forecasts(scfg)
    vb = pd.read_parquet(res / "eval" / "var_backtests.parquet")
    c1 = skew_comparison(scfg, fc)
    v1, _ = verdict(c1, lambda_table(scfg))
    c2 = h2_comparison(scfg, fc, vb)
    v2, l2 = verdict_h2(c2)
    h1 = (
        f"**Placebo check (automatic).** On the synthetic data (symmetric-t innovations, so H1 is false by construction) "
        f"the same rule returns **{v1}**."
    )
    h2 = (
        f"**Placebo check (automatic).** On the synthetic data, where H2 is false by construction, the same rule returns "
        f"**{v2}**: " + " ".join(l2)
    )
    if v2 != "NOT SUPPORTED":
        h2 += (
            " Because the placebo produces the same kind of outcome, the real-data result above cannot be "
            "distinguished from chance. **H2 should be treated as not established.** The flaw is in the rule: "
            "E2 counts one-sided 5% tests across 14 cells with no multiple-testing correction, so one or two "
            "'significant' cells are expected by luck. It is recorded rather than quietly rewritten (DEVELOPMENT_LOG #14)."
        )
    return h1, h2
