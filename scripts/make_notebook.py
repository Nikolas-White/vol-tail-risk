"""(Re)create notebooks/01_exploration.ipynb. Notebooks hold exploration/figures only; logic lives in src/."""

from __future__ import annotations

import nbformat as nbf

from voltail.config import PROJECT_ROOT

cells = [
    nbf.v4.new_markdown_cell(
        "# Exploration: stylised facts of daily returns\n\n"
        "Run `make data` first (or set `CONFIG = 'config/synthetic.yaml'`). This notebook only *looks* at the data; "
        "every computation used in results lives in `src/voltail`."
    ),
    nbf.v4.new_code_cell(
        "import numpy as np, pandas as pd, matplotlib.pyplot as plt\n"
        "from scipy import stats\n"
        "from voltail.config import load_config\n"
        "from voltail.data.store import load_processed\n"
        "CONFIG = None  # or 'config/synthetic.yaml'\n"
        "cfg = load_config(CONFIG)\n"
        "data = load_processed(cfg)\n"
        "list(data)"
    ),
    nbf.v4.new_markdown_cell(
        "## 1. Volatility clustering: returns are ~uncorrelated, squared returns are not"
    ),
    nbf.v4.new_code_cell(
        "def acf(x, lags=50):\n"
        "    x = (x - x.mean()) / x.std()\n"
        "    return [np.mean(x[k:] * x[:-k]) for k in range(1, lags + 1)]\n"
        "fig, axes = plt.subplots(1, 2, figsize=(12, 3.5), sharey=True)\n"
        "for a, df in data.items():\n"
        "    r = df['ret'].to_numpy()\n"
        "    axes[0].plot(acf(r), lw=1, label=a)\n"
        "    axes[1].plot(acf(r**2), lw=1, label=a)\n"
        "axes[0].set_title('ACF of returns'); axes[1].set_title('ACF of squared returns'); axes[1].legend(ncol=2)\n"
        "for ax in axes: ax.axhline(0, color='k', lw=0.6)"
    ),
    nbf.v4.new_markdown_cell(
        "## 2. Fat tails: compare return quantiles with a normal distribution"
    ),
    nbf.v4.new_code_cell(
        "rows = []\n"
        "for a, df in data.items():\n"
        "    r = df['ret']\n"
        "    rows.append({'asset': a, 'ann_vol_%': r.std() * np.sqrt(252), 'skew': stats.skew(r), 'excess_kurtosis': stats.kurtosis(r),\n"
        "                 'worst_day_%': r.min(), 'days_below_-4sd': int((r < r.mean() - 4 * r.std()).sum()),\n"
        "                 'expected_if_normal': len(r) * stats.norm.cdf(-4)})\n"
        "pd.DataFrame(rows).round(3)"
    ),
    nbf.v4.new_markdown_cell(
        "## 3. Leverage effect: do down days raise next-day range more than up days?"
    ),
    nbf.v4.new_code_cell(
        "for a, df in data.items():\n"
        "    nxt = np.log(df['garman_klass'].shift(-1))\n"
        "    q = pd.qcut(df['ret'], 10, labels=False)\n"
        "    plt.plot(nxt.groupby(q).mean().to_numpy(), marker='o', lw=1, label=a)\n"
        "plt.xlabel('decile of today\\'s return (0 = worst)'); plt.ylabel('mean log GK tomorrow'); plt.legend(ncol=2)"
    ),
]
nb = nbf.v4.new_notebook(
    cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}}
)
out = PROJECT_ROOT / "notebooks" / "01_exploration.ipynb"
nbf.write(nb, out)
print(f"wrote {out}")
