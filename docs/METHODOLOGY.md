# Methodology: every design decision and why

This document records what the pipeline does and why each choice was made. Where
there was a judgement call, the more conservative option was taken and the
alternative is noted. Parameters live in `config/default.yaml`; file references
point to where each piece is implemented.

---

## 1. Ground rules

| Rule | How it is enforced |
|---|---|
| No invented numbers | Every number in `README.md`, `RESUME_BULLETS.md` and the `docs/` files sits inside an `<!-- AUTO:... -->` block written by `make report` (`src/voltail/reporting/report.py`, `inject.py`). Hand-written prose never quotes a result. Worked examples in the learning guide come from `scripts/learning_examples.py`. |
| No lookahead | Timing convention in `src/voltail/models/base.py`; checked by the perturbation tests in `tests/test_no_lookahead.py` (see section 5). |
| Parameters estimated only on past data | Refit points and estimation windows are set in `src/voltail/backtest/walkforward.py`. Every forecast row stores the date of the last observation used for estimation (`train_end_<family>`), and a test asserts that this date is always strictly before the forecast date. |
| Hyperparameters tuned only before the test period | XGBoost is tuned once per asset and horizon, on the burn-in window that precedes the first out-of-sample forecast, and the settings are then frozen (`src/voltail/models/ml.py`). The production forecast reuses these frozen settings (`results/xgb_params.json`), so the served model is the validated one. No other model has tuned hyperparameters. EWMA uses lambda = 0.94 and HS uses 250/500-day windows: industry conventions fixed in advance. |
| Determinism | One seed in the config. EGARCH simulation draws are seeded by (config seed, asset, refit date, model); MCS bootstraps and Z2 Monte Carlo by (config seed, asset, level/horizon, regime, model); XGBoost uses the config seed with single-threaded training. Reruns are bit-for-bit identical, and the no-lookahead test would fail if they were not. |

## 2. Data

**Source.** Daily OHLC from Yahoo Finance via `yfinance`, starting 2005-01-01, with
`auto_adjust=True` so that ETF prices (GLD, TLT) include distributions. Raw downloads
are cached as parquet (`data/raw/`). Download time, ticker used and any
substitution are recorded in `data/raw/manifest.json`.

**Universe and the choices behind it.**

| Asset | Ticker (fallback) | Why this instrument |
|---|---|---|
| S&P 500 | `^GSPC` (`SPY`) | Benchmark equity index. |
| S&P/TSX | `^GSPTSE` (`XIU.TO`) | Canadian equity; relevant to Toronto employers. |
| EUR/USD, GBP/USD | `EURUSD=X`, `GBPUSD=X` | Liquid FX with lower volatility and weaker leverage effects. |
| Gold | `GLD` (`GC=F`) | **GLD, not the `GC=F` futures**: continuous futures series contain roll jumps and, in free data, unreliable early highs/lows. The ETF is a cleaner, tradeable proxy. |
| Rates | `TLT` (`IEF`) | **TLT, not the `^TNX` yield**: VaR needs a *price* return. Changes in a yield are not a P&L, and a percentage change of a yield is not a meaningful return when yields are near zero (2020). TLT is a long-duration price proxy for rates risk. |
| Bitcoin | `BTC-USD` | Fat-tail stress case. Trades every day, data only from Sept 2014. |

**Fallbacks.** If a ticker fails after four attempts with exponential back-off, the
fallback is used. If every ticker fails, the asset is skipped and the skip is shown in
the data table of the report. If a run needed a fallback, it appears in the data table
as "(fallback)".

**Incomplete bars.** If the download runs during market hours, Yahoo returns today's
still-forming bar. Rows dated today are dropped before 18:00 New York time (crypto:
anything dated today in UTC). Drops are recorded in the manifest
(`download.py::drop_incomplete_bar`). The scheduled refresh runs at 23:37 UTC, after
18:00 New York all year round.

**Cleaning (`src/voltail/data/clean.py`).** Every dropped or flagged row is counted,
with example dates, in `data/processed/data_quality.json` and in the report.

1. Rows with a missing or non-positive close are dropped.
2. Weekend rows of non-crypto assets are dropped (free FX feeds sometimes publish them).
3. *Stale* rows are dropped: close equal to the previous close **and** high equal to low. Free FX feeds publish these on holidays; they are not trading days, and keeping them would add fake zero returns.
4. *Spike-and-revert bad ticks* are dropped: |r_t| above 8 robust SDs (1.4826 × the median absolute return over the previous **20** days) **and** the next day reverses at least 80% of the move. A genuine crash that does not revert is kept, and so is a genuine whipsaw inside a volatile period (the short window makes the scale adapt). Tests check both. On the real data this flags only five EUR/USD prints in 2008, all dated on the 8th of a month (5.3% to 16% one-day moves that fully reverse), which looks like a day/month date-parsing error in the source.
   *Caveat:* using r_{t+1} means this step looks one day ahead. This mirrors how vendors correct bad prints after the fact. It removes at most a handful of rows, which are counted and dated in the report. Those days are absent from both model inputs and the VaR evaluation sample, and their move is merged into the next day's return. This is the only place where future information touches the data. It is accepted because a 20%+ bad print that reverses the next day is a data error, and keeping it would distort every model more than dropping it. A stricter variant (flag instead of drop) is listed under future work.
5. Rows whose OHLC is internally inconsistent (high below open/close, low above open/close, or a zero range) keep their close-to-close return but get `range_ok = False` and NaN range proxies.

**Calendars.** Each asset keeps its own trading calendar. Prices are never
forward-filled to invent trading days: a holiday simply has no row, and the next return
covers the gap. Cross-asset information is aligned later with *strictly lagged* as-of
joins (section 4.6).

**Returns.** Close-to-close log returns in percent, `r_t = 100 ln(C_t / C_{t-1})`.
Percent units keep GARCH optimisers well conditioned.

## 3. Volatility proxies (`src/voltail/features/proxies.py`)

True daily variance is unobservable. With intraday data one would use realized
variance (the sum of squared 5-minute returns). Free daily data does not allow that,
so three proxies are used:

| Proxy | Formula (percent²) | Pros | Cons |
|---|---|---|---|
| Squared return | r_t² | Conditionally unbiased for close-to-close variance | Very noisy: one draw of a chi-square-like variable |
| Parkinson (1980) | (ln H/L)² / (4 ln 2) | Roughly 5x more efficient than r² under Brownian motion | Misses the overnight gap; biased by discrete sampling and bid-ask bounce |
| Garman-Klass (1980) | 0.5 (ln H/L)² − (2 ln 2 − 1)(ln C/O)² | Roughly 7x more efficient | Same overnight problem; relies on the quality of the open price |

**Why r² is the primary evaluation proxy.** Patton (2011) shows that ranking forecasts
by MSE or QLIKE gives the correct ranking *in expectation* as long as the proxy is
conditionally unbiased, however noisy it is. r² meets that condition. The range proxies
do not, because they omit the overnight return. Using r² therefore keeps the ranking
valid; noise only costs statistical power, which the DM and MCS tests account for.

**Why Garman-Klass is the modelling target and the secondary proxy.** A less noisy
target is better for *estimation* (HAR, XGBoost). To put a range proxy on the
close-to-close scale, it is multiplied by the ratio Σr² / ΣGK, computed on an
**expanding window using only past data** (minimum 250 observations). As an evaluation
proxy, day t's GK is scaled with the ratio as of t−1. All results are reported with
both proxies. When they disagree, that disagreement is itself a finding.

The report's proxy table shows the raw GK/r² ratio (how much the overnight gap
matters for each asset) and first-order autocorrelations. A higher autocorrelation
means less measurement noise, because true variance is persistent and noise is not.

## 4. Models

All models produce, for each origin t (a close) and the next day t+1:
`var1` (variance forecast), `var5` (average daily variance over t+1..t+5), and for
tail models 1-day VaR and ES at 99% and 97.5%. They also store a grid of 40 predictive
tail quantiles, used by the ES backtest. VaR and ES are reported as positive losses in
percent.

### 4.1 Baselines (`src/voltail/models/baselines.py`)
- **Rolling SD** over 21 days (variance of the last 21 returns). Volatility only.
- **EWMA / RiskMetrics**, λ = 0.94 (the 1996 RiskMetrics daily value; not tuned). VaR/ES assume normal returns with zero mean, the RiskMetrics convention. The 5-day forecast equals the 1-day forecast (EWMA has no mean reversion).
- **Historical simulation**, 250-day window (the regulatory minimum of one year). VaR is the empirical quantile (linear interpolation) and ES the mean of returns at or below it. Not used for volatility.

### 4.2 GARCH family via `arch` (`src/voltail/models/garch_arch.py`)
GARCH(1,1) (normal and Student-t), GJR-GARCH(1,1,1) (normal, Student-t and Hansen's
skewed t) and EGARCH(1,1,1) (Student-t), constant mean, `rescale=False`. The skewed-t
ES has no closed form and is computed by numerical integration of the quantile
function (`distributions.py::skewt_es`, checked against the Student-t closed form and
Monte Carlo in `tests/test_distributions.py`).

- **Refit monthly** on an expanding window, warm-started from the previous estimates. Between refits the parameters are fixed and the variance recursion is run forward through new data. This is how such models are used in practice, and it keeps the cost manageable (about 220 refits per model per asset).
- **Convergence failures**: if the optimiser does not converge even after a cold restart, the previous month's parameters are kept and a note is logged (`results/diagnostics.json`). Unconverged estimates are never used silently.
- **Multi-step forecasts**: analytic for GARCH and GJR. EGARCH has no closed form, so its 5-day forecast uses `arch`'s simulation method (1,000 paths) with an explicitly seeded sampler. The 1-step value is taken from the exact analytic recursion.
- **VaR/ES**: VaR = −(μ + σ q_α), ES = −μ + σ ES_α(z), with the normal or a *unit-variance* Student-t (`src/voltail/models/distributions.py`).
- **Implementation subtlety (documented):** when forecasting, `arch` clips variances to bounds computed from the whole residual series passed in. The engine only ever passes data up to the block's last origin, but within a block that includes days after a given origin. The bounds are about 10⁶ times typical variance and have no practical effect. The perturbation test would catch a binding bound only for origins before its cut-off, so this rests on the size of the bounds, not on the test.

### 4.3 GARCH(1,1) from scratch (`src/voltail/models/garch_scratch.py`)
Gaussian MLE with SLSQP under α + β < 1, using the same EWMA backcast as `arch` so the
two can be compared like for like. The variance recursion is evaluated as a linear
filter (`scipy.signal.lfilter`). Standard errors come from a numerical Hessian.
Verified against `arch` in `tests/test_garch_scratch.py` (parameters, log-likelihood,
the variance path to 1e-6, and forecasts to 1e-8). It is also reconciled on every real
asset in `make fit` (report section 7).

### 4.4 HAR (`src/voltail/models/har.py`, `ols.py`)
log-HAR on Garman-Klass: regress log of the mean GK over t+1..t+h on log GK_t, log of
the 5-day mean and log of the 22-day mean. The regression uses OLS from scratch,
solving the normal equations by Cholesky. Newey-West errors are needed because 5-day
targets overlap. Matched against statsmodels in `tests/test_har.py`.

- **Logs** keep forecasts positive and make residuals closer to Gaussian. Back-transforming with exp() underestimates the mean (Jensen's inequality), so forecasts are multiplied by the Duan smearing factor mean(exp(residual)).
- The forecast is in GK units and is converted to close-to-close variance with the past-only scale ratio at the origin.
- **VaR/ES**: zero mean; quantile and tail mean of the *in-sample* standardized returns r_{t+1}/σ̂_t (FHS-style). With 4 parameters and thousands of observations, in-sample overfitting is negligible.
- Refit monthly.

### 4.5 Tail models (`src/voltail/models/tail.py`)
- **Filtered HS (`fhs_gjr`)**: standardized residuals of the GJR-t fit over the estimation window; VaR/ES from their empirical quantile and tail mean, scaled by the GJR forecast σ.
- **Conditional EVT (`evt_gjr`)**, McNeil and Frey (2000): a GPD fitted to standardized losses above their 90th percentile, by maximum likelihood from scratch (Nelder-Mead). VaR and ES use the closed-form POT expressions. ξ is capped at 0.95 (ξ ≥ 1 implies infinite ES), and any cap is logged.
- **Removed in the trim:** a 63-day rolling SD, 500-day HS, EGARCH-normal and unconditional EVT. Each duplicated a model already in the set and added nothing to the conclusions of the first run (DEVELOPMENT_LOG #13).

### 4.6 XGBoost (`src/voltail/models/ml.py`, `src/voltail/features/ml_features.py`)
- **Target**: log mean GK over t+1..t+h (h = 1, 5).
- **Features** (all known at close t): log GK (1/5/22-day means), log Parkinson, log r² (1/5/22-day means), r_t, min(r_t, 0) for the leverage effect, the 5-day return sum, log EWMA variance, and log 63-day rolling variance. Also three S&P 500 features as of the **last S&P date strictly before t** (`merge_asof(..., allow_exact_matches=False)`). Closing times differ: the free FX daily bar ends after the US close, and Bitcoin trades around the clock. A same-date join could therefore leak information, so the full-day lag is the conservative choice.
- **Training rows** are those whose target window ends on or before the refit date. A 5-day target never peeks past the cut-off.
- **Hyperparameters**: a small grid (depth 2–3, 200 or 500 trees, learning rate 0.03, subsample 0.8, min_child_weight 20). It is searched **once per asset and horizon, on the burn-in window only**, with expanding-window CV, a 22-day purge and QLIKE as the score, then frozen. The grid is deliberately small and shallow: a large search on about 750 rows would overfit the CV.
- **Smearing and VaR multiplier from out-of-fold predictions**: tree models fit their training data closely, so in-sample residuals understate forecast error. Both are computed from blocked K-fold out-of-fold predictions inside the training window, with 22-day purge gaps.
- Refit quarterly (cost), `n_jobs=1` and a fixed seed for determinism.

**Why no LSTM/MLP.** It is the first item in the agreed cut order. A neural network
on about 4,000 daily observations per asset is unlikely to beat HAR or GARCH, and it
would add a heavy dependency (PyTorch) and non-determinism concerns. That compute time
was better spent on the MCS and the ES test. It is listed as future work.

## 5. Walk-forward engine (`src/voltail/backtest/walkforward.py`)

- **Burn-in**: the first 750 observations of each asset (about 3 years) are estimation-only. With data from 2005, out-of-sample forecasts start in late 2007 or early 2008 for most assets, so the 2008 crisis is **out of sample**. Bitcoin (daily data from September 2014, 7 days a week) starts in October 2016. ML tuning therefore happens on the calm 2005–2007 period, which is a disclosed limitation.
- **Refit points**: the last trading day of each month (GARCH family, tail models, HAR) or quarter (XGBoost), plus the end of the burn-in.
- **Expanding window** by default (stable parameters, maximum data). A rolling window is available in the config (`walkforward.window: rolling`). It adapts faster to regime change at the cost of noisier estimates; this sensitivity is listed as future work.
- **Timing**: a block of origins [R, R_next − 1] uses parameters estimated at R. The model receives only rows up to the block's last origin. The forecast made at origin o is stored against the date of row o+1.
- **No-lookahead test** (`tests/test_no_lookahead.py`): for every model family, the engine is run twice: on the original data, and on a copy where every return, range and price after a cut-off date (mid-block) is randomly rescaled. The cross-asset driver is perturbed from the cut-off date *inclusive*, to test the strict lag. Every forecast up to and including the first day after the cut-off must match to 1e-10. A second assertion checks that later forecasts *do* change, so the test cannot pass vacuously. This test caught a real problem during development (DEVELOPMENT_LOG #2).

## 6. Evaluation (`src/voltail/evaluation/`)

**Common sample.** Every comparison uses the dates on which *every* model in the
comparison has a forecast and the proxy is observed.

### 6.1 Volatility
- **QLIKE** (Patton form proxy/f + ln f; it differs from the normalised form only by a proxy-only constant, so rankings and tests are identical, and it stays finite on zero-return days). QLIKE is preferred because it is robust to proxy noise and is not dominated by a few extreme days the way MSE is. MSE is reported as well.
- **Mincer-Zarnowitz**: proxy = a + b·forecast, with a HAC Wald test of (a, b) = (0, 1). R² against r² is low by construction.
- **Diebold-Mariano** against GARCH(1,1)-normal and pairwise, with Newey-West variance (lags ≥ h − 1) and the Harvey-Leybourne-Newbold correction.
- **Model Confidence Set** (Hansen, Lunde and Nason 2011), 90% level, stationary block bootstrap with 1,000 replications, via `arch.bootstrap.MCS`. Being *in* the MCS means "not statistically worse than the best". Rankings are always reported alongside MCS membership to avoid over-claiming.
- Next-week forecasts are evaluated against the mean proxy over t..t+4.

### 6.2 VaR / ES
- **Kupiec POF** (correct number of exceedances), **Christoffersen independence** (no clustering) and **conditional coverage** (both together).
- **Basel traffic light**: the BCBS zones (0–4 green, 5–9 yellow, 10+ red per 250 days at 99%) are defined by where the Binomial(250, 1%) CDF crosses 95% and 99.99%. The same rule is applied to the full out-of-sample sample and to 97.5%; the report calls these "Basel-style" zones. The share of rolling 250-day windows in each zone is also reported at 99%.
- **ES: Acerbi-Szekely Z2** with Monte Carlo p-values (5,000 simulations). Under H0, each simulated day draws U ~ U(0,1). If U < α, its loss is the model's own predictive quantile at U, interpolated from the stored 40-point log-spaced grid in [1e-4, 0.025], with a power-law tail below 1e-4. This works the same way for parametric, HS, FHS and EVT models. The Z2 test was chosen over McNeil-Frey because McNeil-Frey needs a σ_t, which historical simulation does not have.
- **FZ0** (Fissler-Ziegel) joint VaR/ES loss: average ranks plus an MCS on FZ0.

### 6.3 Regimes
GFC 2008-01-01 to 2009-12-31; COVID 2020-02-15 to 2020-06-30; 2022 (the whole year);
Calm = 2013–2014, 2017 and 2019. These are fixed calendar windows chosen in advance.
They are used only to *slice evaluation*, never as model inputs. Reported for each
regime: QLIKE ranks, exceedances against expectation, Kupiec, and the number of
exceedances in the first 20 trading days of each window, which is where slow adapters
such as HS fail.

### 6.4 Ratings (`src/voltail/evaluation/ratings.py`)
A VaR/ES cell (model × asset × level) **passes** if Kupiec p ≥ 0.05, Christoffersen CC
p ≥ 0.05, Z2 p ≥ 0.05, and the full-sample zone is not red. With s the pass share:
**Satisfactory** if s ≥ 0.8 and there is no red zone; **Satisfactory with findings**
if 0.5 ≤ s < 0.8 and there is no red zone; **Needs improvement** if 0.25 ≤ s and
(s < 0.5 or there is a red zone); **Unsatisfactory** if s < 0.25. The rule was written
before any real-data run, so the ratings cannot be tuned to the results.

**Multiple testing.** Hundreds of tests are run. At a 5% level, roughly 1 in 20
correct-model cells will fail by chance, which is why ratings use pass *shares* across
assets and levels rather than any single p-value.

## 7. Deviations from the original specification

| Spec | What was done | Why |
|---|---|---|
| Data 2005 to present | Done. The first 750 days are burn-in. | Gives about 3 years of estimation data before the first forecast while keeping the 2008 crisis out of sample. |
| Gold `GC=F` or `GLD` | `GLD` (with `GC=F` as fallback) | Futures roll artefacts (section 2). |
| 10y yield or TLT | `TLT` | A yield is not a price; VaR needs P&L. |
| LSTM/MLP optional | Cut | First item in the agreed cut order (section 4.6). |
| Acerbi-Szekely Z2 **or** McNeil-Frey | Z2 | Works for every model, including HS (section 6.2). |
| Deploy to a free host | Instructions only (`docs/DEPLOYMENT.md`) | Deploying requires the owner's account; the Docker image is host-agnostic. |
| Results in the docs | AUTO blocks, empty until `make all` runs with internet access | The build sandbox could not reach Yahoo Finance (DEVELOPMENT_LOG #1). The whole pipeline was validated end to end on synthetic data with known parameters (`reports/synthetic/RESULTS.md`). |

## 8. Run metadata

<!-- AUTO:meta -->
_Auto-generated by `make report` on 2026-10-07 20:59 UTC_  
_Data source: **yfinance** (downloaded 2026-10-07T16:41:45+00:00); out-of-sample forecasts through 2026-10-06; git commit `n/a`; config hash `8d54d4d52b7a`; seed 20260101._
<!-- /AUTO:meta -->

## 9. Pre-registered research question (written 2026-10-07, before the skewed-t model was run on real data)

**Observation from the first real-data run.** GJR-GARCH-t ranked near the top for
volatility, yet its 99% VaR was exceeded on about 1.6% (S&P 500) and 1.8% (TSX) of
days, which puts it in the Basel red zone. Filtered historical simulation, which uses
the *same* GJR volatility but the empirical distribution of past standardized
residuals, was close to 1%. The volatility forecast was therefore not the problem;
the assumed *shape* of the return distribution was. Equity index returns are
negatively skewed, and a symmetric Student-t cannot represent that.

**Hypothesis H1.** Replacing the symmetric Student-t with Hansen's (1994) skewed
Student-t (`gjr_skewt`; same GJR variance equation, one extra parameter λ) fixes the
equity tail.

**Decision rule (applied mechanically by `src/voltail/reporting/research.py`).** For each
of the S&P 500 and TSX:

- **C1:** full-sample λ < 0 with |λ/SE| > 2 (statistically significant negative skew).
- **C2:** at 99%, the skewed-t model passes Kupiec (p ≥ 0.05) and is not in the red zone.
- **C3:** at 99%, the skewed-t model has lower FZ0 loss than the symmetric-t model, with a one-sided Diebold-Mariano p < 0.05.

**SUPPORTED** if C1–C3 hold for both indices. **PARTIALLY SUPPORTED** if C1 holds for
both and C2 or C3 holds for at least one. **NOT SUPPORTED** otherwise. The result is
reported whichever way it comes out (README, RESULTS.md section 0, validation report
section 7.4). The other five assets are shown for context, including any where skewness
makes things worse.

**Placebo.** `make synthetic` applies the same rule to data simulated with symmetric-t
innovations, where H1 is false by construction. The rule should return NOT SUPPORTED there.

**What else changed at the same time** (so the comparison stays honest): the bad-wick
data fix (DEVELOPMENT_LOG #12) and trimming the model list (roll63, hs500, EGARCH-normal
and unconditional EVT removed as redundant). Both models in H1 are re-run on the same
cleaned data, so the comparison between them is like for like.

<!-- AUTO:research -->
**Question.** GJR-GARCH-t forecast equity volatility well but breached its 99% VaR too often (red zone for the S&P 500 and TSX in the first run), while filtered HS with the same volatility did not. Is the failure caused by ignoring negative skewness? The hypothesis and the decision rule were written down before the skewed-t model was run (METHODOLOGY section 9, `src/voltail/reporting/research.py`).

**Verdict (mechanical, pre-registered rule): SUPPORTED.**

- SPX: C1 met (lambda = -0.163, t = -9.1); C2 met (99% exceedances 1.59% -> 1.21%, zone red -> green, Kupiec p 0.166); C3 met (FZ0 difference -0.0451, one-sided DM p 0.005).
- TSX: C1 met (lambda = -0.242, t = -12.5); C2 met (99% exceedances 1.78% -> 0.98%, zone red -> green, Kupiec p 0.870); C3 met (FZ0 difference -0.0425, one-sided DM p 0.022).

Estimated skewness (lambda < 0 = heavier left tail):

| asset   |   lambda_full_sample |    se |   t_stat |   lambda_refit_min |   lambda_refit_median |   lambda_refit_max |
|:--------|---------------------:|------:|---------:|-------------------:|----------------------:|-------------------:|
| SPX     |               -0.163 | 0.018 |   -9.111 |             -0.178 |                -0.159 |             -0.139 |
| TSX     |               -0.242 | 0.019 |  -12.474 |             -0.330 |                -0.245 |             -0.234 |
| EURUSD  |               -0.006 | 0.018 |   -0.309 |             -0.044 |                -0.020 |              0.061 |
| GBPUSD  |               -0.026 | 0.018 |   -1.436 |             -0.036 |                -0.022 |              0.062 |
| GOLD    |               -0.064 | 0.018 |   -3.596 |             -0.107 |                -0.070 |             -0.052 |
| UST10   |               -0.058 | 0.021 |   -2.821 |             -0.077 |                -0.059 |              0.001 |
| BTC     |               -0.001 | 0.017 |   -0.046 |             -0.059 |                -0.022 |              0.013 |

GJR-GARCH-t vs GJR-GARCH-skewed-t, every asset and level:

| asset   | level   |    n | exceed t   |   Kupiec p (t) | zone t   | exceed skew-t   |   Kupiec p (skew-t) | zone skew-t   |   FZ0 skew-t minus t |   DM p (skew-t better) |
|:--------|:--------|-----:|:-----------|---------------:|:---------|:----------------|--------------------:|:--------------|---------------------:|-----------------------:|
| SPX     | 99.0%   | 4723 | 1.59%      |          0.000 | red      | 1.21%           |               0.166 | green         |               -0.045 |                  0.005 |
| SPX     | 97.5%   | 4723 | 3.66%      |          0.000 | red      | 2.75%           |               0.274 | green         |               -0.034 |                  0.001 |
| TSX     | 99.0%   | 4711 | 1.78%      |          0.000 | red      | 0.98%           |               0.870 | green         |               -0.042 |                  0.022 |
| TSX     | 97.5%   | 4711 | 3.74%      |          0.000 | red      | 2.48%           |               0.942 | green         |               -0.030 |                  0.010 |
| EURUSD  | 99.0%   | 4888 | 1.04%      |          0.762 | green    | 0.98%           |               0.899 | green         |                0.002 |                  0.744 |
| EURUSD  | 97.5%   | 4888 | 2.41%      |          0.699 | green    | 2.43%           |               0.768 | green         |                0.001 |                  0.830 |
| GBPUSD  | 99.0%   | 4905 | 1.33%      |          0.029 | yellow   | 1.26%           |               0.074 | yellow        |                0.003 |                  0.831 |
| GBPUSD  | 97.5%   | 4905 | 3.20%      |          0.003 | yellow   | 3.12%           |               0.007 | yellow        |                0.000 |                  0.589 |
| GOLD    | 99.0%   | 4723 | 1.14%      |          0.333 | green    | 1.10%           |               0.492 | green         |               -0.010 |                  0.164 |
| GOLD    | 97.5%   | 4723 | 2.90%      |          0.085 | yellow   | 2.52%           |               0.931 | green         |               -0.008 |                  0.108 |
| UST10   | 99.0%   | 4723 | 1.14%      |          0.333 | green    | 0.97%           |               0.857 | green         |               -0.003 |                  0.287 |
| UST10   | 97.5%   | 4723 | 2.99%      |          0.038 | yellow   | 2.71%           |               0.361 | green         |               -0.004 |                  0.097 |
| BTC     | 99.0%   | 3652 | 1.40%      |          0.023 | yellow   | 1.31%           |               0.069 | yellow        |               -0.014 |                  0.031 |
| BTC     | 97.5%   | 3652 | 3.50%      |          0.000 | red      | 3.45%           |               0.000 | yellow        |               -0.011 |                  0.008 |

**Placebo check (automatic).** On the synthetic data (symmetric-t innovations, so H1 is false by construction) the same rule returns **NOT SUPPORTED**.
<!-- /AUTO:research -->


## 10. Second pre-registered question (written 2026-10-07, before the skew-filtered tail models were run)

**Motivation.** Section 9's result showed that the symmetric-t GJR model misreads the
equity left tail. The two semi-parametric tail models (filtered HS and conditional EVT)
still use that symmetric-t GJR model as their volatility *filter*.

**Hypothesis H2.** Filtering with GJR-skewed-t instead of GJR-t improves the
semi-parametric tail models. New models: `fhs_skewt` and `evt_skewt`, identical to
`fhs_gjr` / `evt_gjr` except for the filter.

**Prior expectation, stated honestly:** probably *little* change. FHS and EVT already
take the tail shape from the empirical standardized residuals, so skewness is largely
captured whatever the filter. The filter mainly changes σ_t, and the GJR-t and skewed-t
volatility paths are very similar. H2 is a fair test exactly because it could fail.

**Decision rule** (`src/voltail/reporting/research.py::verdict_h2`), for each pair
(fhs_gjr → fhs_skewt, evt_gjr → evt_skewt):

- **E1:** the number of asset × level cells passing all backtests (the rating rule: Kupiec, Christoffersen CC, Z2 ≥ 0.05, not red) is strictly higher with the skewed-t filter.
- **E2:** across the 14 asset × level cells, the number where the skewed-t filter has significantly lower FZ0 loss (one-sided DM p < 0.05) exceeds the number where it is significantly worse.

**SUPPORTED** if E1 and E2 hold for both pairs; **PARTIALLY SUPPORTED** if E1 or E2 holds
for at least one pair and neither pair gets worse on E1; **NOT SUPPORTED** otherwise.
Reported whichever way it comes out (README and RESULTS.md section 0).

<!-- AUTO:research_h2 -->
**Question.** Does filtering the semi-parametric tail models (filtered HS, conditional EVT) with the skewed-t GJR model instead of the symmetric-t one improve them? Pre-registered in METHODOLOGY section 10, with the stated prior that it would change little (these models already take the tail shape from the data).

**Verdict (mechanical, pre-registered rule): PARTIALLY SUPPORTED.**

- fhs_gjr -> fhs_skewt: E1 NOT met (cells passing all tests 8 -> 8 of 14); E2 met (FZ0 significantly better in 2 cells, significantly worse in 0).
- evt_gjr -> evt_skewt: E1 NOT met (cells passing all tests 11 -> 11 of 14); E2 met (FZ0 significantly better in 2 cells, significantly worse in 0).

| pair                 | asset   | level   | exceed (t filter)   | exceed (skew-t filter)   | passes (t filter)   | passes (skew-t filter)   |   FZ0 new minus old |   DM p (skew-t filter better) |
|:---------------------|:--------|:--------|:--------------------|:-------------------------|:--------------------|:-------------------------|--------------------:|------------------------------:|
| fhs_gjr -> fhs_skewt | SPX     | 99.0%   | 1.16%               | 1.19%                    | False               | False                    |              -0.004 |                         0.005 |
| fhs_gjr -> fhs_skewt | SPX     | 97.5%   | 2.48%               | 2.50%                    | True                | True                     |              -0.002 |                         0.018 |
| fhs_gjr -> fhs_skewt | TSX     | 99.0%   | 1.02%               | 1.04%                    | True                | True                     |               0.002 |                         0.829 |
| fhs_gjr -> fhs_skewt | TSX     | 97.5%   | 2.44%               | 2.42%                    | True                | True                     |               0.003 |                         0.949 |
| fhs_gjr -> fhs_skewt | EURUSD  | 99.0%   | 1.04%               | 1.04%                    | True                | True                     |               0.000 |                         0.754 |
| fhs_gjr -> fhs_skewt | EURUSD  | 97.5%   | 2.50%               | 2.50%                    | True                | True                     |               0.000 |                         0.608 |
| fhs_gjr -> fhs_skewt | GBPUSD  | 99.0%   | 1.18%               | 1.16%                    | False               | False                    |               0.000 |                         0.693 |
| fhs_gjr -> fhs_skewt | GBPUSD  | 97.5%   | 2.98%               | 2.98%                    | False               | False                    |              -0.000 |                         0.292 |
| fhs_gjr -> fhs_skewt | GOLD    | 99.0%   | 1.08%               | 1.08%                    | True                | True                     |              -0.002 |                         0.072 |
| fhs_gjr -> fhs_skewt | GOLD    | 97.5%   | 2.41%               | 2.41%                    | False               | False                    |              -0.001 |                         0.085 |
| fhs_gjr -> fhs_skewt | UST10   | 99.0%   | 1.33%               | 1.33%                    | False               | False                    |              -0.001 |                         0.164 |
| fhs_gjr -> fhs_skewt | UST10   | 97.5%   | 3.01%               | 3.03%                    | False               | False                    |               0.000 |                         0.523 |
| fhs_gjr -> fhs_skewt | BTC     | 99.0%   | 0.79%               | 0.79%                    | True                | True                     |              -0.001 |                         0.182 |
| fhs_gjr -> fhs_skewt | BTC     | 97.5%   | 2.41%               | 2.41%                    | True                | True                     |               0.000 |                         0.896 |
| evt_gjr -> evt_skewt | SPX     | 99.0%   | 1.08%               | 1.10%                    | True                | True                     |              -0.004 |                         0.003 |
| evt_gjr -> evt_skewt | SPX     | 97.5%   | 2.58%               | 2.60%                    | True                | True                     |              -0.002 |                         0.020 |
| evt_gjr -> evt_skewt | TSX     | 99.0%   | 0.93%               | 0.96%                    | True                | True                     |               0.003 |                         0.893 |
| evt_gjr -> evt_skewt | TSX     | 97.5%   | 2.38%               | 2.33%                    | True                | True                     |               0.002 |                         0.939 |
| evt_gjr -> evt_skewt | EURUSD  | 99.0%   | 0.94%               | 0.96%                    | True                | True                     |              -0.000 |                         0.387 |
| evt_gjr -> evt_skewt | EURUSD  | 97.5%   | 2.37%               | 2.37%                    | True                | True                     |               0.000 |                         0.707 |
| evt_gjr -> evt_skewt | GBPUSD  | 99.0%   | 1.14%               | 1.14%                    | True                | True                     |               0.000 |                         0.602 |
| evt_gjr -> evt_skewt | GBPUSD  | 97.5%   | 2.98%               | 2.98%                    | False               | False                    |              -0.000 |                         0.272 |
| evt_gjr -> evt_skewt | GOLD    | 99.0%   | 0.93%               | 0.93%                    | True                | True                     |              -0.001 |                         0.181 |
| evt_gjr -> evt_skewt | GOLD    | 97.5%   | 2.31%               | 2.31%                    | False               | False                    |              -0.001 |                         0.096 |
| evt_gjr -> evt_skewt | UST10   | 99.0%   | 1.08%               | 1.08%                    | True                | True                     |              -0.000 |                         0.376 |
| evt_gjr -> evt_skewt | UST10   | 97.5%   | 2.88%               | 2.92%                    | False               | False                    |              -0.000 |                         0.248 |
| evt_gjr -> evt_skewt | BTC     | 99.0%   | 0.74%               | 0.74%                    | True                | True                     |               0.001 |                         0.929 |
| evt_gjr -> evt_skewt | BTC     | 97.5%   | 2.35%               | 2.35%                    | True                | True                     |               0.000 |                         0.800 |

**Placebo check (automatic).** On the synthetic data, where H2 is false by construction, the same rule returns **PARTIALLY SUPPORTED**: fhs_gjr -> fhs_skewt: E1 NOT met (cells passing all tests 10 -> 10 of 14); E2 met (FZ0 significantly better in 2 cells, significantly worse in 0). evt_gjr -> evt_skewt: E1 NOT met (cells passing all tests 12 -> 12 of 14); E2 met (FZ0 significantly better in 2 cells, significantly worse in 0). Because the placebo produces the same kind of outcome, the real-data result above cannot be distinguished from chance. **H2 should be treated as not established.** The flaw is in the rule: E2 counts one-sided 5% tests across 14 cells with no multiple-testing correction, so one or two 'significant' cells are expected by luck. It is recorded rather than quietly rewritten (DEVELOPMENT_LOG #14).
<!-- /AUTO:research_h2 -->
