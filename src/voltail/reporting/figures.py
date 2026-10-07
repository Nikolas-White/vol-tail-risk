"""Static figures for RESULTS.md / README (matplotlib, PNG).

Colour roles follow a validated categorical palette (fixed slot order, never
cycled), a single-hue sequential ramp for magnitudes, and reserved status colours
(pass / warning / fail) that always come with a text label.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from voltail.evaluation.regimes import regime_mask  # noqa: E402
from voltail.models.registry import MODEL_LABELS  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e1e0d9"
NEUTRAL = "#9a9890"
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)
REGIME_FILL = "#f0efec"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "font.size": 9.5,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.6,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def _ann_factor(asset_class: str) -> int:
    return 365 if asset_class == "crypto" else 252


def _shade_regimes(
    ax, index: pd.DatetimeIndex, regimes: dict[str, Any], label: bool = True
) -> None:
    for name, spec in regimes.items():
        m = regime_mask(index, spec)
        if not m.any():
            continue
        pos = np.flatnonzero(m)
        starts = np.concatenate([[0], np.flatnonzero(np.diff(pos) > 1) + 1])
        ends = np.concatenate([starts[1:] - 1, [len(pos) - 1]])
        for s, e in zip(starts, ends, strict=True):
            ax.axvspan(index[pos[s]], index[pos[e]], color=REGIME_FILL, zorder=0, lw=0)
            if label:
                ax.text(
                    index[pos[s]],
                    1.0,
                    f" {name}",
                    transform=ax.get_xaxis_transform(),
                    va="top",
                    ha="left",
                    fontsize=8,
                    color=INK_2,
                )


def _banner(fig, cfg: dict[str, Any]) -> None:
    if cfg["data"]["source"] == "synthetic":
        fig.text(
            0.99,
            0.005,
            "SYNTHETIC DATA - not market results",
            ha="right",
            va="bottom",
            fontsize=8,
            color=STATUS["critical"],
        )


def fig_vol_forecasts(fc: pd.DataFrame, asset: str, cfg: dict[str, Any], out: Path) -> Path:
    _style()
    ann = _ann_factor(cfg["data"]["assets"][asset]["asset_class"])
    fig, ax = plt.subplots(figsize=(11, 4.2))
    proxy = np.sqrt(ann * fc["rv1_gk"].rolling(21, min_periods=10).mean())
    ax.plot(
        proxy.index, proxy, color=NEUTRAL, lw=1.2, label="Realized proxy (Garman-Klass, 21d avg)"
    )
    for i, m in enumerate([m for m in ["gjr_t", "har", "xgb"] if f"var1_{m}" in fc]):
        s = np.sqrt(ann * fc[f"var1_{m}"])
        ax.plot(s.index, s, color=SERIES[i], lw=1.1, label=f"{MODEL_LABELS[m]} forecast")
    _shade_regimes(ax, fc.index, cfg["evaluation"]["regimes"])
    ax.set_ylabel("annualized volatility, %")
    ax.set_title(
        f"{asset}: next-day volatility forecasts vs realized proxy (out-of-sample)", loc="left"
    )
    ax.legend(loc="upper right", ncol=2, fontsize=8.5)
    _banner(fig, cfg)
    path = out / "fig1_vol_forecasts.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_var_exceedances(
    fc: pd.DataFrame, asset: str, cfg: dict[str, Any], out: Path, models=("gjr_t", "hs250")
) -> Path:
    _style()
    lab = "a010"
    models = [m for m in models if f"VaR_{lab}_{m}" in fc]
    zoom_spec = cfg["evaluation"]["regimes"].get("COVID") or [
        str(fc.index[-250].date()),
        str(fc.index[-1].date()),
    ]
    zoom = regime_mask(fc.index, zoom_spec)
    pad = np.flatnonzero(zoom)
    zsel = slice(max(0, pad[0] - 30), min(len(fc), pad[-1] + 30)) if len(pad) else slice(-250, None)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [2.2, 1]})
    for ax, sel, title in [
        (axes[0], slice(None), "full out-of-sample period"),
        (axes[1], zsel, "zoom"),
    ]:
        d = fc.iloc[sel]
        ax.vlines(d.index, 0, d["ret"], color=NEUTRAL, lw=0.5, label="daily return")
        for i, m in enumerate(models):
            v = -d[f"VaR_{lab}_{m}"]
            ax.plot(d.index, v, color=SERIES[i], lw=1.1, label=f"-VaR 99% {MODEL_LABELS[m]}")
            hit = d["ret"] < v
            ax.scatter(
                d.index[hit],
                d["ret"][hit],
                s=22,
                color=SERIES[i],
                edgecolor=SURFACE,
                linewidth=1.0,
                zorder=3,
                label=f"exceedance ({int(hit.sum())})",
            )
        ax.set_title(f"{asset} 99% VaR: {title}", loc="left")
        ax.set_ylabel("return, %")
    loc = mdates.AutoDateLocator(minticks=3, maxticks=5)
    axes[1].xaxis.set_major_locator(loc)
    axes[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(0.01, -0.08),
        ncol=len(labels),
        fontsize=8.5,
    )
    _banner(fig, cfg)
    path = out / "fig2_var_exceedances.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_qlike_rank_heatmap(
    vm: pd.DataFrame, mcs: pd.DataFrame, cfg: dict[str, Any], out: Path, horizon: int = 1
) -> Path:
    _style()
    proxy = cfg["evaluation"]["primary_vol_proxy"]
    sel = vm[(vm["horizon"] == horizon) & (vm["proxy"] == proxy) & (vm["regime"] == "Full")]
    piv = sel.pivot(index="model", columns="asset", values="rank_qlike")
    order = piv.mean(axis=1).sort_values().index
    piv = piv.loc[order, [a for a in cfg["data"]["assets"] if a in piv.columns]]
    m = (
        mcs[(mcs["horizon"] == horizon) & (mcs["proxy"] == proxy)]
        .pivot(index="model", columns="asset", values="in_mcs")
        .reindex_like(piv)
    )
    fig, ax = plt.subplots(figsize=(1.2 + 0.9 * piv.shape[1], 0.45 * piv.shape[0] + 1.4))
    ax.grid(False)
    ax.imshow(piv.to_numpy(), cmap=SEQ.reversed(), aspect="auto", vmin=1, vmax=len(piv))
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            r = piv.iat[i, j]
            dark = r <= len(piv) / 2
            txt = f"{r:.0f}" + (" *" if bool(m.iat[i, j]) else "")
            ax.text(
                j, i, txt, ha="center", va="center", fontsize=8.5, color="white" if dark else INK
            )
    ax.set_xticks(range(piv.shape[1]), piv.columns)
    ax.set_yticks(range(piv.shape[0]), [MODEL_LABELS.get(x, x) for x in piv.index])
    ax.set_title(
        f"QLIKE rank by asset ({'next-day' if horizon == 1 else 'next-week'}, proxy: r^2)\n1 = best;  * = in 90% Model Confidence Set",
        loc="left",
        fontsize=10.5,
    )
    _banner(fig, cfg)
    path = out / f"fig3_qlike_rank_h{horizon}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_var_backtest_grid(
    vb: pd.DataFrame, cfg: dict[str, Any], out: Path, alpha: float = 0.01
) -> Path:
    _style()
    sel = vb[(vb["regime"] == "Full") & (np.isclose(vb["alpha"], alpha))].copy()

    def status(r) -> tuple[str, str]:
        if r["zone"] == "red" or r["p_pof"] < 0.01:
            return STATUS["critical"], "fail"
        if r["p_pof"] < 0.05 or r["p_cc"] < 0.05 or r["p_z2"] < 0.05:
            return STATUS["warning"], "flag"
        return STATUS["good"], "pass"

    sel[["color", "word"]] = sel.apply(lambda r: pd.Series(status(r)), axis=1)
    models = (
        sel.groupby("model")["word"]
        .apply(lambda w: (w == "pass").mean())
        .sort_values(ascending=False)
        .index
    )
    assets = [a for a in cfg["data"]["assets"] if a in set(sel["asset"])]
    fig, ax = plt.subplots(figsize=(1.6 + 1.05 * len(assets), 0.42 * len(models) + 1.6))
    ax.grid(False)
    for i, m in enumerate(models):
        for j, a in enumerate(assets):
            r = sel[(sel["model"] == m) & (sel["asset"] == a)]
            if r.empty:
                continue
            r = r.iloc[0]
            ax.add_patch(plt.Rectangle((j - 0.47, i - 0.42), 0.94, 0.84, color=r["color"], lw=0))
            ax.text(
                j,
                i,
                f"{100 * r['rate']:.2f}% {r['word']}",
                ha="center",
                va="center",
                fontsize=7.8,
                color="white" if r["word"] == "fail" else INK,
            )
    ax.set_xlim(-0.5, len(assets) - 0.5)
    ax.set_ylim(len(models) - 0.5, -0.5)
    ax.set_xticks(range(len(assets)), assets)
    ax.set_yticks(range(len(models)), [MODEL_LABELS.get(x, x) for x in models])
    ax.set_title(
        f"{1 - alpha:.1%} VaR backtests: exceedance rate (target {alpha:.1%})\n"
        "pass = Kupiec, Christoffersen CC and Z2 all p>=0.05, not red;  flag = one test p<0.05;  fail = red zone or Kupiec p<0.01",
        loc="left",
        fontsize=9.5,
    )
    _banner(fig, cfg)
    path = out / f"fig4_var_backtests_{round((1 - alpha) * 1000)}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_regime_exceedances(
    vb: pd.DataFrame, cfg: dict[str, Any], out: Path, alpha: float = 0.01
) -> Path:
    _style()
    sel = vb[np.isclose(vb["alpha"], alpha)]
    present = set(sel["regime"])
    regs = [r for r in ["Full", *cfg["evaluation"]["regimes"]] if r in present]
    agg = sel.groupby(["regime", "model"])[["exceed", "expected"]].sum()
    agg["ratio"] = agg["exceed"] / agg["expected"]
    order = agg.loc["Full"].sort_values("ratio").index if "Full" in regs else agg.index.levels[1]
    fig, axes = plt.subplots(
        1, len(regs), figsize=(2.3 * len(regs) + 2.2, 0.32 * len(order) + 1.8), sharey=True
    )
    axes = np.atleast_1d(axes)
    for ax, reg in zip(axes, regs, strict=True):
        d = agg.loc[reg].reindex(order)
        ax.barh(range(len(d)), d["ratio"], color=SERIES[0], height=0.62)
        ax.axvline(1.0, color=INK, lw=1)
        ax.set_title(reg, loc="left", fontsize=10)
        ax.set_xlabel("exceedances / expected")
        for i, (ratio, x) in enumerate(zip(d["ratio"], d["exceed"], strict=True)):
            ax.text(ratio, i, f" {ratio:.1f}x ({int(x)})", va="center", fontsize=7, color=INK_2)
        ax.set_xlim(0, max(3.0, float(np.nanmax(agg["ratio"])) * 1.35))
    axes[0].set_yticks(range(len(order)), [MODEL_LABELS.get(x, x) for x in order])
    fig.suptitle(
        f"{1 - alpha:.0%} VaR exceedances by regime, pooled across assets (1.0x = perfectly calibrated)",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    _banner(fig, cfg)
    path = out / "fig5_regime_exceedances.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_proxy_comparison(fc: pd.DataFrame, asset: str, cfg: dict[str, Any], out: Path) -> Path:
    _style()
    ann = _ann_factor(cfg["data"]["assets"][asset]["asset_class"])
    fig, ax = plt.subplots(figsize=(11, 3.8))
    for i, (col, label) in enumerate(
        [
            ("rv1_sq", "squared return"),
            ("rv1_pk", "Parkinson (scaled)"),
            ("rv1_gk", "Garman-Klass (scaled)"),
        ]
    ):
        s = np.sqrt(ann * fc[col].rolling(5, min_periods=3).mean())
        ax.plot(s.index, s, color=SERIES[i], lw=0.9, label=label, alpha=0.9)
    ax.set_ylabel("annualized vol, % (5-day avg of proxy)")
    ax.set_title(
        f"{asset}: three noisy measurements of the same unobservable volatility", loc="left"
    )
    ax.legend(loc="upper right")
    _banner(fig, cfg)
    path = out / "fig6_proxy_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_param_stability(params: pd.DataFrame, cfg: dict[str, Any], out: Path) -> Path:
    _style()
    p = params[params["model"] == "gjr_t"].pivot_table(
        index=["asset", "refit_date"], columns="param", values="value"
    )
    p["persistence"] = p["alpha[1]"] + 0.5 * p["gamma[1]"] + p["beta[1]"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    assets = [a for a in cfg["data"]["assets"] if a in p.index.get_level_values(0)]
    for i, a in enumerate(assets):
        d = p.loc[a]
        axes[0].plot(d.index, d["persistence"], color=SERIES[i % 8], lw=1.3, label=a)
        axes[1].plot(d.index, d["nu"], color=SERIES[i % 8], lw=1.3, label=a)
    axes[0].set_title(
        "GJR-GARCH-t persistence (a + g/2 + b) at each refit", loc="left", fontsize=10.5
    )
    axes[1].set_title(
        "Student-t degrees of freedom (lower = fatter tails)", loc="left", fontsize=10.5
    )
    axes[1].legend(ncol=4, fontsize=8)
    _banner(fig, cfg)
    path = out / "fig7_param_stability.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_xgb_importance(params: pd.DataFrame, cfg: dict[str, Any], out: Path) -> Path | None:
    _style()
    imp = params[params["model"] == "xgb_importance"]
    if imp.empty:
        return None
    s = imp.groupby("param")["value"].mean().sort_values().tail(12)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(range(len(s)), s.to_numpy(), color=SERIES[0], height=0.62)
    ax.set_yticks(range(len(s)), s.index)
    ax.set_xlabel("mean share of total gain (all assets, all refits)")
    ax.set_title("XGBoost (next-day): what the trees actually use", loc="left")
    _banner(fig, cfg)
    path = out / "fig8_xgb_importance.png"
    fig.savefig(path)
    plt.close(fig)
    return path
