"""Rule-based model ratings for the validation report.

Ratings are produced mechanically from backtest outcomes so that no judgement
call in the report can be tweaked after seeing the results. The rules are stated
here and reproduced verbatim in docs/MODEL_VALIDATION_REPORT.md.

Per (model, asset, alpha) a VaR/ES forecast **passes** when all of:
  * Kupiec POF p-value >= 0.05
  * Christoffersen conditional-coverage p-value >= 0.05
  * Acerbi-Szekely Z2 p-value >= 0.05
  * full-sample Basel-style zone is not red

Model rating from the pass share s across all (asset, alpha) cells:
  * Satisfactory                  s >= 0.80 and no red zone anywhere
  * Satisfactory with findings    0.50 <= s < 0.80 and no red zone
  * Needs improvement             0.25 <= s, or any red zone with s >= 0.25
  * Unsatisfactory                s < 0.25
"""

from __future__ import annotations

import pandas as pd

RATING_ORDER = ["Satisfactory", "Satisfactory with findings", "Needs improvement", "Unsatisfactory"]


def cell_pass(row: pd.Series, level: float = 0.05) -> bool:
    return bool(
        row["p_pof"] >= level
        and row["p_cc"] >= level
        and row["p_z2"] >= level
        and row["zone"] != "red"
    )


def rate_models(var_full: pd.DataFrame) -> pd.DataFrame:
    """``var_full``: full-sample VaR backtest rows (asset, alpha, model, p_pof, p_cc, p_z2, zone)."""
    df = var_full.copy()
    df["pass"] = df.apply(cell_pass, axis=1)
    rows = []
    for model, g in df.groupby("model", sort=False):
        s = float(g["pass"].mean())
        any_red = bool((g["zone"] == "red").any())
        if s < 0.25:
            rating = "Unsatisfactory"
        elif any_red or s < 0.5:
            rating = "Needs improvement"
        elif s < 0.8:
            rating = "Satisfactory with findings"
        else:
            rating = "Satisfactory"
        failed = g.loc[~g["pass"], ["asset", "alpha"]]
        rows.append(
            {
                "model": model,
                "cells": len(g),
                "passed": int(g["pass"].sum()),
                "pass_share": s,
                "red_zones": int((g["zone"] == "red").sum()),
                "rating": rating,
                "failed_cells": ", ".join(
                    f"{a}@{1 - al:.1%}" for a, al in failed.itertuples(index=False)
                ),
            }
        )
    out = pd.DataFrame(rows)
    out["order"] = out["rating"].map(RATING_ORDER.index)
    return (
        out.sort_values(["order", "pass_share"], ascending=[True, False])
        .drop(columns="order")
        .reset_index(drop=True)
    )
