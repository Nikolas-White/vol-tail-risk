# Learning guide: every concept in this project, in plain English

Written for someone who codes well but has not taken a formal ML, statistics or
linear algebra course. Each section gives the intuition, a small worked example, the
one-line "interview version", and **where it lives in the code**. All worked-example
numbers are produced by `python scripts/learning_examples.py` (output saved in
`docs/learning_examples_output.md`), so you can change the inputs and watch them move.

**Suggested study order (about 2 weeks at 1–2 hours a day):** 1 → 2 → 3 → 5 → 4 → 6
→ 7 → 8 → 9 → 10 → 11 → 12 → 13 → 14 → 15 → 16 → 17 → 18 → 19 → 20. Then read
`docs/METHODOLOGY.md` end to end and work through `docs/INTERVIEW_QA.md` out loud.

---

## 0. Units and conventions used everywhere

- A **return** is `r_t = 100 × ln(P_t / P_{t-1})`, a percent log return. Log returns add up over time, which is convenient.
- **Volatility** σ is the standard deviation of returns, quoted *per day* in this repo. To annualize, multiply by √252 (trading days) or √365 for Bitcoin. A daily vol of 1% is about 15.9% a year.
- **Variance** is σ². Models forecast variance; we take square roots for display.
- **VaR and ES are positive loss numbers.** "99% VaR = 2.5" means a 1% chance of losing more than 2.5% tomorrow.

## 1. Volatility clustering

**Intuition.** Calm days follow calm days and wild days follow wild days. Returns
themselves are almost unpredictable: today's sign says nearly nothing about tomorrow's.
But the *size* of returns is predictable. Plot the autocorrelation of r and of r²
(the notebook does this): r is about 0 at every lag, while r² is positive and decays
slowly over weeks.

**Why it matters.** It is the reason volatility forecasting works at all. Every model
here except rolling SD and HS exploits it explicitly.

**Interview version.** "Returns are close to unpredictable, but their magnitude is
persistent. GARCH-type models turn yesterday's shock and yesterday's variance into
tomorrow's variance."

**Code.** `notebooks/01_exploration.ipynb` (ACF plots).

## 2. EWMA / RiskMetrics

**Idea.** Tomorrow's variance is a weighted average of today's variance estimate and
today's squared return:

`σ²_{t+1} = λ σ²_t + (1 − λ) r_t²`, with λ = 0.94.

The weights on past squared returns decay geometrically: 6% on today, 5.6% on
yesterday, and so on. JP Morgan's RiskMetrics (1996) fixed λ = 0.94 for daily data.

**Worked example.** σ_t = 1.00%, today r_t = −3.00%:
σ²_{t+1} = 0.94×1 + 0.06×9 = **1.4800**, so σ_{t+1} = **1.2166%**. One big move raises
tomorrow's vol by about 22%.

**Weakness.** No long-run mean. EWMA's forecast for next week equals its forecast for
tomorrow, so after a spike it never "expects" volatility to come back down. It is
GARCH with ω = 0 and α + β = 1 (called IGARCH).

**Code.** `src/voltail/models/baselines.py::ewma_variance` (implemented as a linear filter).

## 3. GARCH(1,1)

**Idea.** EWMA plus a pull toward a long-run level:

`σ²_{t+1} = ω + α ε_t² + β σ²_t`, where ε_t = r_t − μ.

- α: how strongly today's surprise moves tomorrow's variance (the "news" reaction).
- β: how much of today's variance carries over (memory).
- α + β < 1: **persistence**. Shocks fade, and variance mean-reverts to **ω / (1 − α − β)**.
- **Half-life** of a shock: ln(0.5) / ln(α + β) days.

**Worked example.** ω=0.02, α=0.08, β=0.90, σ²_t=1, r_t=−3 (μ=0):
σ²_{t+1} = 0.02 + 0.08×9 + 0.90×1 = **1.6400** (σ = 1.2806%). Long-run variance =
0.02/0.02 = **1.0000** (15.87% annualized). Half-life = **34.31 days**. Forecasts decay
toward 1: h=1: 1.6400, h=2: 1.6272, h=5: 1.5903, h=10: 1.5336.
Multi-step formula: `σ²_{t+h} = σ̄² + (α+β)^{h−1} (σ²_{t+1} − σ̄²)`.

**Assumptions to know for interviews.** Parameters are constant over time. The
response is symmetric (a −3% day and a +3% day have the same effect). The innovation
distribution is fixed (normal or t). Covariance stationarity requires α + β < 1.

**Code.** `src/voltail/models/garch_scratch.py` (from scratch), `garch_arch.py` (library, used in the backtest).

## 4. Maximum likelihood estimation (MLE)

**Idea.** Pick the parameters that make the observed data most probable. For each
candidate (μ, ω, α, β), run the variance recursion and compute the probability density
of each day's return under N(μ, σ²_t). Multiply across days; in practice add the logs,
which is the **log-likelihood**. Then let an optimiser search for the maximum.

`log L = −½ Σ_t [ ln(2π) + ln σ²_t + ε_t² / σ²_t ]`

The two terms trade off: a small σ²_t rewards you through the ln σ² term but punishes
you heavily when a big ε_t arrives (ε²/σ²).

**Worked example (one parameter).** Data −1, 0.5, 2, −0.5 from N(0, σ²). Log-likelihoods:
σ=0.8: −7.0801; σ=1.0: −6.4258; **σ=1.1726: −6.3127 (the maximum)**; σ=1.3: −6.3524;
σ=1.6: −6.6300. The maximum is at √(mean x²) = 1.1726, the closed-form answer for this
simple case. GARCH has no closed form, hence `scipy.optimize`.

**Standard errors** come from the curvature of the log-likelihood at the peak (the
Hessian). A sharp peak means a precisely estimated parameter.

**Code.** `garch_scratch.py::gaussian_nll`, `fit_garch11` (SLSQP with the α+β<1 constraint and a numerical Hessian). The test `tests/test_garch_scratch.py` shows it matches `arch` (on the 6,000-day test sample the estimates agree to 5 decimals; the test tolerance is 2e-3).

## 5. Asymmetry and the leverage effect: GJR-GARCH and EGARCH

**Fact.** For equities, a −3% day raises future volatility more than a +3% day. (The
"leverage" story: falling prices raise firms' debt/equity ratio. Fear and
deleveraging also play a role.)

**GJR-GARCH** adds a term that only switches on after negative shocks:
`σ²_{t+1} = ω + (α + γ·1[ε_t<0]) ε_t² + β σ²_t`.
Worked example: α=0.03, γ=0.10, β=0.90, ω=0.02, σ²_t=1: after +3% → **1.1900**; after
−3% → **2.0900**, which is **75.6% higher**.

**EGARCH** models *log* variance, so positivity is automatic and no parameter
constraints are needed: `ln σ²_{t+1} = ω + α(|z_t| − E|z|) + γ z_t + β ln σ²_t`. A
negative γ gives the leverage effect.

**Code.** `garch_arch.py::SPECS`. Persistence for GJR is α + γ/2 + β (because γ is
active half the time for symmetric shocks).

## 6. Fat tails and the Student-t distribution

Real daily returns have more extreme days than a normal distribution predicts (excess
kurtosis; see the notebook's "days below −4 sd" against the normal expectation). A
Student-t with ν degrees of freedom has fatter tails: the lower ν, the fatter. We use
a *unit-variance* t, so σ still means the standard deviation.

**Worked example.** σ = 1.5%. Normal 99% VaR = 2.326 × 1.5 = **3.4895%**, ES =
**3.9978%**. Unit-variance t(4): VaR = **3.9742%**, ES = **5.5373%**. Same volatility,
much bigger tail.

**Code.** `src/voltail/models/distributions.py`.

## 6b. Skewness and the skewed Student-t

A symmetric t has fat tails on *both* sides equally. Equity indices crash harder than
they rally (negative skew), so a symmetric distribution fitted to equity returns puts
too little probability in the left tail, and VaR is exceeded too often. Hansen's
(1994) skewed t adds one parameter λ in (−1, 1): λ < 0 fattens the left tail, and λ = 0
is the ordinary t. Its ES has no neat formula, so the code integrates the quantile
function numerically: ES = −(1/α) ∫₀^α q(u) du.

This is the project's research question. The symmetric-t GJR failed at 99% on the S&P
500 and TSX, while FHS (same volatility, empirical shape) did not. Does adding skew
fix it? The test was written down before running it (METHODOLOGY section 9); the
answer is in the README.

**Code.** `distributions.py::skewt_quantile`, `skewt_es`; `reporting/research.py`.

## 7. HAR (Heterogeneous Autoregressive) model

**Idea (Corsi 2009).** Market participants work on different horizons: day traders,
weekly rebalancers, monthly funds. Tomorrow's volatility is a linear combination of
yesterday's, last week's average and last month's average:

`log v_{t+1} = b0 + b_d log v_t + b_w log mean(v_{t−4..t}) + b_m log mean(v_{t−21..t}) + u`

where v is a daily variance measure (here Garman-Klass, section 9). It is just a linear
regression, which makes it fast, transparent and hard to overfit, yet it is a famously
tough benchmark.

**Smearing.** We fit in logs. Because E[exp(u)] > 1 (Jensen's inequality),
exponentiating the fitted log value *underestimates* the mean. Multiply by
mean(exp(residuals)) (Duan's smearing estimator).

**Code.** `src/voltail/models/har.py`.

## 8. OLS: the closed-form solution (mini linear-algebra primer)

**Setup.** Stack the regressors into a matrix X (one row per day, one column per
regressor, with a column of 1s for the intercept) and the targets into a vector y. We
want coefficients b that minimise the sum of squared errors Σ(y − Xb)².

**Matrix basics you need.** Xᵀ is X flipped (rows become columns). XᵀX is a small
k×k matrix (k = number of regressors), and (XᵀX)⁻¹ is its inverse, the matrix
version of "1/x". Setting the derivative of the squared-error sum to zero gives the
**normal equations** XᵀX b = Xᵀy, so

**b = (XᵀX)⁻¹ Xᵀ y**.

**Worked example.** X = [[1,1],[1,2],[1,3]], y = [2, 2.5, 4.5] → b = **[0.5000,
1.2500]**: intercept 0.5, slope 1.25. Check: the line passes through the mean point
(2, 3).

**In code** we never compute the inverse directly (numerically fragile). We solve the
normal equations with a Cholesky factorisation, which gives the same answer more
stably. **Standard errors**: classical s²(XᵀX)⁻¹. **Newey-West (HAC)** errors fix the
classical ones when residuals are autocorrelated, which they always are for
overlapping 5-day targets.

**Code.** `src/voltail/models/ols.py`; verified against statsmodels in `tests/test_har.py`.

## 9. Volatility proxies: Parkinson, Garman-Klass, squared returns

We cannot observe true daily variance. Options from daily data:
- **r²**: unbiased (its average equals the variance) but extremely noisy. It is one random draw.
- **Parkinson**: uses the high–low range, which "sees" the whole day's path. About 5x more informative than r² for Brownian motion.
- **Garman-Klass**: adds open and close. About 7x more informative.

The catch: highs and lows only cover *trading hours*, so range estimators miss the
overnight jump from yesterday's close to today's open and are biased low. We rescale by
the historical ratio mean(r²)/mean(GK), computed using past data only.

**Interview version.** "Without intraday data I used range-based estimators for
modelling because they are much less noisy, but I evaluated with r² because QLIKE
rankings are only guaranteed to be correct with an unbiased proxy."

**Code.** `src/voltail/features/proxies.py`; test `test_range_proxies_unbiased_for_brownian_motion`.

## 10. Loss functions: QLIKE vs MSE

To score a variance forecast f against a proxy p:
- **MSE** = (p − f)². Symmetric, and dominated by a few huge days, since crisis variances are 100x calm ones and get squared.
- **QLIKE** = p/f − ln(p/f) − 1 (≥ 0; zero when f = p). It looks at the *ratio* p/f, so calm and wild days count on similar scales. It penalises *under*-prediction more, which is what a risk manager fears.

**Worked example.** True variance 4. Forecast 2 (half too low): QLIKE = **0.3069**,
MSE = 4.0. Forecast 6 (half too high): QLIKE = **0.0721**, MSE = 4.0. MSE treats both
errors the same; QLIKE says under-forecasting is about 4x worse.

**Patton (2011) robustness.** If the proxy is unbiased (like r²), ranking models by
average QLIKE or MSE gives the same ranking as if you knew the true variance, *in
expectation*. Many other losses do not have this property. That is why QLIKE plus r²
is the primary metric. (The code uses the equivalent form p/f + ln f, which stays
finite when p = 0.)

**Code.** `src/voltail/evaluation/losses.py`; test `test_qlike_minimised_at_true_variance_with_noisy_proxy`.

## 11. Mincer-Zarnowitz regressions

Regress the proxy on the forecast: p_t = a + b f_t + e. A perfect forecast has a = 0
and b = 1. b < 1 means forecasts swing too much; a > 0 with b < 1 means they are too
flat. We test a = 0, b = 1 jointly with robust errors. With r², the R² is always low
because r² is so noisy. Compare R² *across models*, not with 1.

**Code.** `src/voltail/evaluation/forecast_tests.py::mincer_zarnowitz`.

## 12. Diebold-Mariano test

**Question.** Model A has a lower average loss than model B. Is that real or luck?

Take the daily loss difference d_t = L_A − L_B. If the models are equally good, the
mean of d is 0. DM = mean(d) / standard error(mean(d)). The standard error must allow
for autocorrelation in d (Newey-West), because loss differences are serially
correlated, especially for 5-day forecasts. We add the Harvey-Leybourne-Newbold
small-sample correction.

**Worked example.** 500 days, model 2 is worse by 0.05 on average plus noise (sd 0.5)
→ mean difference **−0.0881**, DM = **−4.1973**, p = **0.0000**. Model 1 is
significantly better. (The realised mean difference is not exactly −0.05 because of
the noise.)

**Code.** `forecast_tests.py::diebold_mariano`.

## 13. Model Confidence Set (MCS)

DM compares *two* models. With 10 models, 45 pairwise tests produce false
"significant" results by chance. The MCS (Hansen, Lunde and Nason 2011) works the other
way round: start with all models, repeatedly test "are they all equally good?" with a
bootstrap, and eliminate the worst until the test no longer rejects. What survives is a
set that contains the best model with 90% confidence.

**How to talk about it** (template; fill in from your results): "Model X ranked first,
but N other models were in the 90% MCS, so I would not claim X is better than them,
only that none of them is detectably worse."

**Code.** `forecast_tests.py::model_confidence_set` (wraps `arch.bootstrap.MCS`).

## 14. VaR vs Expected Shortfall

- **VaR_99%**: the loss you exceed only 1% of the time. It says nothing about *how bad* those 1% days are.
- **ES_99%** (also CVaR): the *average* loss on those worst 1% days. It looks into the tail.

**Why Basel moved from 99% VaR to 97.5% ES (FRTB).** (1) VaR ignores tail severity: two
portfolios with the same VaR can have very different crash losses. (2) VaR is not
*subadditive*: in some cases VaR(A+B) > VaR(A) + VaR(B), so it can penalise
diversification. ES is coherent. (3) 97.5% was chosen because, for a normal
distribution, 97.5% ES is about equal to 99% VaR (normal ES_97.5% = **3.5067%** vs
VaR_99% = **3.4895%** at σ = 1.5%), so the capital level stayed comparable while
becoming tail-sensitive. **Downside:** ES is harder to backtest. It is not
"elicitable" on its own, so there is no simple scoring rule for ES alone.

**Code.** All models' `VaR_*` / `ES_*` columns; `distributions.py`.

## 15. Historical simulation (HS)

Take the last 250 (or 500) daily returns, sort them, and read off the 1% worst: that
is the VaR. ES is the average of the returns beyond it. No model, no distribution
assumption, and very popular in banks. **Problem:** it treats a calm year and a crisis
year the same until crisis days *enter* the window, so it reacts late, then stays high
for a year after the crisis ends (the "ghost effect").

**Code.** `baselines.py::hs_es`.

## 16. Filtered Historical Simulation (FHS)

Best of both worlds. Use GARCH to get *today's* volatility, and use history for the
*shape* of the distribution. Divide each past return by its GARCH volatility on that
day to get standardized residuals z (unit-free shocks). Tomorrow's VaR =
−(μ + σ_{t+1} × 1% quantile of z). GARCH handles "how volatile is it now", and history
handles "how fat is the tail, how skewed".

**Code.** `src/voltail/models/tail.py::fhs_risk`.

## 17. Extreme Value Theory (EVT): peaks over threshold and the GPD

**Problem.** The 1% quantile of 750 observations rests on about 7 data points, and the
0.1% quantile on less than 1. HS cannot extrapolate.

**Theorem (Pickands–Balkema–de Haan).** For almost any distribution, the *excesses over
a high threshold* u (how far beyond u the big losses go) follow a **Generalized Pareto
Distribution** with shape ξ (tail heaviness: ξ > 0 means fat, power-law tails) and
scale β. So fit a GPD to the largest 10% of losses and use its formula for far
quantiles:

VaR_p = u + (β/ξ)[(p / P(loss > u))^(−ξ) − 1],  ES_p = VaR_p/(1−ξ) + (β − ξu)/(1−ξ).

**Worked example.** u = 1.5 (10% of losses exceed it), ξ = 0.2, β = 0.6 → 99% quantile
= **3.2547**, ES = **4.4433**.

**Conditional EVT (McNeil–Frey).** Apply this to GARCH standardized residuals (like
FHS) so the tail model also reacts to current volatility. Without that filter, an EVT
model on raw returns ignores volatility clustering: its exceptions bunch up in crises.

**Code.** `tail.py::fit_gpd` (MLE from scratch), `evt_risk`.

## 18. VaR backtests: Kupiec, Christoffersen, Basel traffic light

Let a **hit** be a day when the loss exceeded VaR. A correct 99% VaR model produces
hits that are (i) about 1% of days and (ii) not clustered.

- **Kupiec POF test** checks (i): is the hit *rate* consistent with 1%? It is a likelihood ratio comparing Bernoulli(observed rate) with Bernoulli(1%), compared with a chi-square(1).
- **Christoffersen independence test** checks (ii): is P(hit today | hit yesterday) the same as P(hit today | no hit yesterday)? Clustered hits mean the model reacts too slowly.
- **Conditional coverage** = both together (chi-square(2)).

**Worked example.** 250 days, 7 hits (three on consecutive days); 2.5 expected.
Kupiec LR = **5.4970**, p = **0.0190** → too many. Christoffersen LR = **6.7362**,
p = **0.0094** → clustered.

**Basel traffic light (1996).** At 99% over 250 days: 0–4 exceptions green, 5–9 yellow
(capital multiplier rises), 10+ red (model presumed flawed). Where do the cut-offs come
from? If the model is right, exceptions ~ Binomial(250, 1%), and P(≤4) = **0.8922**,
P(≤5) = **0.9588**, P(≤9) = **0.9997**, P(≤10) = **0.9999**. Yellow starts where the
cumulative probability passes 95%, red where it passes 99.99%. Seven exceptions are
**yellow**.

**Code.** `src/voltail/evaluation/var_tests.py`.

## 19. ES backtests: Acerbi–Szekely Z2 and FZ0

**Z2.** On hit days, compare the realised loss with the model's ES:
`Z2 = 1 − (1/(T α)) Σ L_t·hit_t / ES_t`. Expected value 0 if the model is right;
negative if losses are bigger or more frequent than the ES implies. The p-value comes
from simulating Z2 many times from the model's *own* predictive distribution.

**FZ0 loss (Fissler–Ziegel).** ES alone has no proper scoring rule, but the pair
(VaR, ES) does. FZ0 is a loss function that is minimised, on average, by the true VaR
and ES. That lets us *rank* tail models, and even run DM or MCS on them. Worked
example (α = 2.5%, VaR = 2, ES = 2.5): a −1% day scores **0.7163**; a −3% day (an
exceedance) scores **16.7163**.

**Code.** `src/voltail/evaluation/es_tests.py`, `losses.py::fz0`.

## 20. Walk-forward evaluation and lookahead bias

**Lookahead bias** means using information you would not have had at the time. It is
the number one reason backtests look better than reality. Sneaky versions: normalising
features with full-sample statistics; tuning hyperparameters on the test period;
aligning assets whose markets close at different times; a 5-day target whose window
extends past the training cut-off; library functions that use the whole series
internally.

**Walk-forward.** At each refit date, fit only on data up to that date, forecast the
following month one day at a time, roll forward, and repeat.

**How this repo *proves* there is no lookahead.** Run the whole engine twice: once on
the real data, once with all data after a cut-off date scrambled. Every forecast made
before the cut-off must be identical. If any piece of code peeks ahead, the numbers
differ and the test fails. It caught a real issue during development (the EGARCH
simulation was not seeded, so reruns differed).

**Code.** `src/voltail/backtest/walkforward.py`, `tests/test_no_lookahead.py`.

## 21. XGBoost for volatility (and why it often does not win)

Gradient-boosted trees learn non-linear relationships from many lagged features.
Volatility is *mostly* a smooth, persistent process that HAR and GARCH already
capture. The remaining signal is small relative to the noise, and trees tend to fit
noise. So in the volatility literature, ML usually improves little over HAR, and only
with richer data (intraday measures, implied volatility, macro news). Whatever this
project's results show (see the README), the honest framing is: "Did the extra
complexity buy statistically significant accuracy (DM/MCS), and was it worth the
opacity for a risk model?"

**Code.** `src/voltail/models/ml.py`.

---

### Concept → file map

| Concept | File |
|---|---|
| Data download, cleaning | `src/voltail/data/download.py`, `clean.py` |
| Proxies | `src/voltail/features/proxies.py` |
| EWMA, rolling SD, HS | `src/voltail/models/baselines.py` |
| GARCH from scratch, MLE | `src/voltail/models/garch_scratch.py` |
| GARCH/GJR/EGARCH (library) | `src/voltail/models/garch_arch.py` |
| HAR, OLS | `src/voltail/models/har.py`, `ols.py` |
| t / normal quantiles, ES, GPD formulas | `src/voltail/models/distributions.py` |
| FHS, EVT | `src/voltail/models/tail.py` |
| XGBoost, features | `src/voltail/models/ml.py`, `src/voltail/features/ml_features.py` |
| Walk-forward engine | `src/voltail/backtest/walkforward.py` |
| QLIKE, MSE, FZ0 | `src/voltail/evaluation/losses.py` |
| MZ, DM, MCS | `src/voltail/evaluation/forecast_tests.py` |
| Kupiec, Christoffersen, Basel | `src/voltail/evaluation/var_tests.py` |
| Z2 | `src/voltail/evaluation/es_tests.py` |
| Regimes, ratings | `src/voltail/evaluation/regimes.py`, `ratings.py` |
| Report generation | `src/voltail/reporting/` |
| API | `src/voltail/api/` |

### Further reading (short list)
- Hull, *Options, Futures and Other Derivatives*: the chapters on volatility and VaR (EWMA, GARCH, HS).
- Christoffersen, *Elements of Financial Risk Management*: the best single book for this project.
- McNeil, Frey and Embrechts, *Quantitative Risk Management*: EVT, ES, backtesting.
- Patton (2011), "Volatility forecast comparison using imperfect volatility proxies", *Journal of Econometrics*.
- Corsi (2009), "A simple approximate long-memory model of realized volatility" (HAR).
- BCBS (1996), "Supervisory framework for the use of backtesting…" (traffic light), and BCBS FRTB (2019) on ES.
