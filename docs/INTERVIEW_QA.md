# Interview Q&A: 40 questions about this project

Answers are written in the first person, as you would say them. They are grounded
in what the code actually does. **Where an answer depends on the results, it points
to the generated numbers instead of stating them.** Read the block below and
`reports/RESULTS.md` before every interview, and never quote a number you cannot
point to.

### Key numbers from the latest real-data run
<!-- AUTO:resume_numbers -->
- assets: 7 (SPX, TSX, EURUSD, GBPUSD, GOLD, UST10, BTC)
- out-of-sample forecast-days (all assets): 32,325; longest history: 18.9 years
- models compared: 15
- best next-day volatility model by average QLIKE rank: GJR-GARCH-normal (3.14)
- best next-week volatility model by average QLIKE rank: GJR-GARCH-normal (3.43)
- XGBoost median QLIKE difference vs GARCH(1,1)-normal (next-day, Garman-Klass proxy): -7.3% (negative = better)
- 99% VaR cells passing Kupiec: 52/84; passing Christoffersen CC: 55/84
- best joint VaR/ES model by average FZ0 rank: XGBoost on lagged features
- models rated Satisfactory: none
<!-- /AUTO:resume_numbers -->

---

## A. The project in general

**1. Walk me through the project in two minutes.**
I wanted to answer a validation question rather than a modelling one: when you
forecast volatility and tail risk honestly, out of sample, which models actually hold
up? I took seven free daily series (S&P 500, TSX, EUR/USD, GBP/USD, gold, long
Treasuries and Bitcoin) from 2005. I ran 13 models in a walk-forward backtest that
refits periodically (monthly for GARCH and HAR, quarterly for XGBoost) using only past data: rolling SD, EWMA, historical simulation, six
GARCH variants (including a skewed-t GJR), HAR, XGBoost, filtered HS and conditional EVT. I scored volatility with
QLIKE, Diebold-Mariano and a Model Confidence Set, and VaR/ES with Kupiec,
Christoffersen, Basel traffic lights, the Acerbi-Szekely ES test and the FZ0 loss,
overall and in 2008, COVID, 2022 and calm periods. Then I wrote it up as a bank-style
validation report, tested one pre-registered hypothesis (does modelling skewness fix
the equity VaR failures?) and served the forecasts through a Dockerized FastAPI service.

**2. What was the main finding?**
I state it straight from the key-numbers block and the README results table: which
model ranked best on average QLIKE, *how many models were statistically
indistinguishable from it* (the MCS), which tail models passed all backtests, and
whether XGBoost added anything. The framing I use: "the ranking is less important than
how few differences are statistically significant."

**3. What would you do differently with more time?**
Intraday realized variance as the target and proxy (it removes most of the proxy noise
problem), a portfolio model (DCC-GARCH or filtered HS on the portfolio P&L),
sensitivity runs for estimation window, EVT threshold and refit frequency, and implied
volatility (VIX) as a feature.

**4. Why did you write a validation report instead of just a README?**
Because in a bank the person who builds a risk model is not the one who signs it off.
Model risk management (SR 11-7 in the US, E-23 in Canada) expects conceptual
soundness, implementation verification, outcome analysis and ongoing monitoring. I
wanted to show I know what that document looks like and can be critical of my own
models.

## B. Data and proxies

**5. Why not use realized volatility?**
Realized volatility needs intraday prices, and free sources only give daily OHLC. So I
used proxies: squared returns, which are unbiased but noisy, and the Parkinson and
Garman-Klass range estimators, which are about 5–7x more efficient but miss the
overnight gap. I modelled on Garman-Klass and evaluated on r², because evaluation needs
an unbiased proxy for the rankings to be valid.

**6. How did you handle the overnight gap in range estimators?**
I rescaled GK to close-to-close variance using the ratio Σr²/ΣGK on an expanding window
of *past* data only. The report shows the raw ratio for each asset. It is a direct
measure of how much variance happens overnight.

**7. How did you handle missing data and holidays?**
Each asset keeps its own calendar; I never forward-fill prices. I drop weekend rows for
non-crypto assets, "stale" rows (unchanged close with high = low, which FX feeds
publish on holidays), and isolated spike-and-revert bad ticks, and every drop is
logged with dates. Bad OHLC days keep their return but lose their range proxy.

**8. The spike filter uses the next day's return. Isn't that lookahead?**
Yes, technically, and I disclose it. It mimics a vendor correcting a bad print the next
day. It only removes moves larger than 12 robust SDs that reverse at least 80%, it
touches a handful of rows (all listed in the report), and a test makes sure a genuine
crash that does not revert is kept.

**9. Why GLD and TLT rather than gold futures and the 10-year yield?**
Continuous futures have roll jumps and poor free high/low data. A yield is not a price,
so changes in it are not P&L, and percentage changes of a yield near zero are
meaningless. TLT gives a price return with rates exposure.

**10. How did you align assets that close at different times?**
For the cross-asset features in XGBoost, I join the S&P 500 as of the last date
*strictly before* the target date, using `merge_asof` with `allow_exact_matches=False`.
The free FX bar closes after the US close and Bitcoin trades 24/7, so a same-day join
could leak. A unit test checks this.

## C. Models

**11. Explain GARCH(1,1) and its assumptions.**
σ²_{t+1} = ω + α ε²_t + β σ²_t. Today's surprise and today's variance determine
tomorrow's variance, mean-reverting to ω/(1−α−β) when α+β<1. The assumptions are
constant parameters, a symmetric response to shocks, a fixed innovation distribution
and covariance stationarity. Typical daily equity estimates have α+β close to 1, so
shocks have half-lives of weeks.

**12. You implemented GARCH from scratch. How did you verify it?**
Gaussian MLE with SLSQP and the constraint α+β<1, using the same backcast initialisation
as `arch`. The tests check that parameters and log-likelihood match `arch`, that
plugging `arch`'s parameters into my filter reproduces its variance path to 1e-6, that
forecasts match to 1e-8, and that simulated data recovers the true parameters within 3
standard errors. `make fit` reconciles it on every real asset too (report section 7).

**13. What is the leverage effect, and how do you model it?**
Negative returns raise future volatility more than positive returns of the same size,
especially in equities. GJR-GARCH adds γ·1[ε<0]·ε²; EGARCH models log variance with an
asymmetric term. In my worked example, next-day variance after a −3% day is about 76%
higher than after a +3% day (2.09 vs 1.19).

**14. Why Student-t innovations, and why did you end up adding a skewed t?**
Daily returns are fat-tailed. Even after GARCH filtering, standardized residuals have
excess kurtosis, and a normal puts the 1% quantile too close to the mean. But the
symmetric t still failed at 99% on the S&P 500 and TSX, while filtered HS with the
*same* volatility forecast didn't. That pointed at the distribution's shape, not the
volatility. Equity returns are negatively skewed, so I wrote down a hypothesis and a
pass/fail rule *before* running a skewed-t version, then ran it. The result, and the
fact that the skew parameter is near zero for FX (where skewed-t changed nothing), is in
the README's research section. That contrast is the part I'd lead with.
I then pre-registered a follow-up, using the skewed-t model as the filter for filtered HS and EVT. My stated prior was that it would barely matter, because those models already take the tail shape from the data. That's what happened: a small FZ0 gain on the S&P 500, no change in pass rates. A null result that matches the prior is still a result.

**15. What is HAR and why is it a strong benchmark?**
A linear regression of future volatility on daily, weekly and monthly averages of past
volatility. It mimics long memory with three parameters, is estimated by OLS, and is
hard to beat in the realized-volatility literature. I fit it in logs with a smearing
correction.

**16. Why did you implement OLS yourself if statsmodels exists?**
To show I understand the closed form b = (XᵀX)⁻¹Xᵀy and the Newey-West correction. I
solve the normal equations by Cholesky rather than inverting, and a test matches
statsmodels to 1e-8, including the HAC errors.

**17. How did you avoid overfitting XGBoost?**
The hyperparameters were tuned once per asset and horizon, on the burn-in window
only, then frozen. The production forecast reuses the same frozen settings. The grid was small and shallow: depth 2–3, min_child_weight 20. I used
expanding-window CV with a 22-day purge, and targets that never extend past the
training cut-off. The smearing factor and the VaR multiplier come from out-of-fold
predictions, because in-sample residuals of a tree model are too small.

**18. Did XGBoost beat GARCH/HAR? Why or why not?**
The answer is in the key-numbers block and section 9 of the report. The general reason
ML struggles here: daily volatility is a smooth, persistent process that GARCH and HAR
already capture, the remaining signal is small relative to proxy noise, and trees fit
noise. If it had won by a wide margin, my first step would have been to audit for
lookahead, which is exactly why the perturbation test exists.

**19. Explain filtered historical simulation.**
GARCH supplies today's volatility; history supplies the shape of the standardized
shocks. VaR = −(μ + σ_{t+1} × the empirical 1% quantile of past z). It reacts like
GARCH but does not assume normal or t tails.

**20. Explain EVT and the GPD.**
By the Pickands–Balkema–de Haan theorem, excesses over a high threshold are
approximately Generalized Pareto, whatever the parent distribution. I fit a GPD by
maximum likelihood to the top 10% of standardized losses and use closed-form formulas
for VaR and ES. The shape ξ measures tail heaviness. I cap ξ at 0.95, because ξ ≥ 1
implies infinite ES. I apply it to GARCH-filtered residuals (conditional EVT), so the tail
model also reacts to the current volatility level.

## D. Evaluation and statistics

**21. Why QLIKE rather than MSE?**
Both are robust to an unbiased noisy proxy in the Patton (2011) sense, but MSE is
dominated by a few crisis days because variance errors get squared. QLIKE works on the
ratio proxy/forecast, so every day counts comparably, and it penalises
under-forecasting more, which is the error that hurts in risk.

**22. What is a Diebold-Mariano test?**
A t-test on the mean loss difference between two models, using a Newey-West variance
because loss differences are autocorrelated. I added the Harvey-Leybourne-Newbold
small-sample correction. It answers "is A's lower average loss more than luck?"

**23. Why the Model Confidence Set?**
With 10 volatility models there are 45 pairwise DM tests, and some will be
"significant" by chance. The MCS starts from all models and eliminates the worst until
equal performance can no longer be rejected. The survivors contain the best model with
90% confidence. It is my guard against over-claiming a winner.

**24. What does a Mincer-Zarnowitz regression tell you?**
Whether forecasts are unbiased and efficient: regress the proxy on the forecast, and
expect intercept 0 and slope 1. R² is low with r² as the proxy because of noise, so I
compare it across models rather than with 1.

**25. Explain the Kupiec and Christoffersen tests.**
Kupiec tests the *number* of VaR exceptions with a likelihood ratio between the
observed hit rate and α. Christoffersen tests *independence*: whether a hit today makes
a hit tomorrow more likely. Conditional coverage combines them. A model can pass Kupiec
and fail Christoffersen if it has the right number of exceptions but they all arrive
in one crisis month.

**26. What are the Basel traffic-light zones and where do they come from?**
At 99% over 250 days: 0–4 exceptions green, 5–9 yellow, 10+ red. The cut-offs are
where the Binomial(250, 1%) cumulative probability crosses 95% and 99.99%, i.e. how
unlikely that many exceptions would be if the model were correct. Yellow and red raise
the capital multiplier from 3 toward 4.

**27. How do you backtest Expected Shortfall?**
ES is not elicitable on its own, so there is no simple loss for it. I used the
Acerbi-Szekely Z2 statistic, which compares realised losses on exception days with the
predicted ES, with p-values simulated from each model's own predictive tail. I also
used the Fissler-Ziegel FZ0 loss, which scores VaR and ES jointly and lets me rank tail
models and run an MCS on them.

**28. You ran thousands of tests. How do you deal with multiple testing?**
At 5%, about 1 in 20 correct-model cells fail by chance. So I do not draw conclusions
from single p-values. The model ratings use pass *shares* across assets and confidence
levels, and the volatility conclusions use the MCS, which controls the family-wise
error.

## E. Risk-management judgement

**29. Why does VaR fail in crises?**
Three reasons. (1) Models calibrated on calm data react with a lag; historical
simulation in particular only reacts when crisis days enter its window. (2)
Normal-tailed models understate the tail even when volatility is right. (3) Parameters
and correlations shift. My regime tables show exceptions concentrated at the onset of
stress windows (the "first 20 days" column).

**30. Why did Basel replace 99% VaR with 97.5% ES?**
VaR ignores how bad losses are beyond the threshold, and it is not subadditive, so it
can penalise diversification. ES averages the tail and is coherent. 97.5% was chosen so
that, under normality, ES is close to 99% VaR (my worked example: 3.51% vs 3.49% at
σ = 1.5%), keeping capital comparable while making it tail-sensitive.

**31. How would this fail if used live?**
Data errors or late data (free feed), a structural break that monthly refits absorb
slowly, univariate models missing correlation spikes, overnight gaps in illiquid
markets, and operational issues like a failed refresh. My `/health` endpoint exposes
forecast staleness, and the validation report has a monitoring plan with triggers.

**32. What does "no lookahead" mean in your code, and how do you prove it?**
A forecast for day t+1 uses only data available at the close of day t, and parameters
are estimated on data before the forecast date. I prove it empirically: run the full
engine twice, once with all data after a cut-off scrambled, and require every earlier
forecast to match to 1e-10. That test caught a real non-determinism bug in an EGARCH
simulation path.

**33. Expanding or rolling window?**
I used expanding by default: more data and more stable parameters. Rolling windows
adapt faster to regime change but are noisier. It is a config switch, and the
sensitivity run is listed as a finding in my validation report.

**34. What if your results looked too good?**
Assume a bug first. An ML model beating GARCH/HAR by a wide margin on QLIKE, or zero
VaR exceptions, would make me audit for lookahead: feature alignment, target windows,
normalisation, tuning periods. My perturbation test automates the most common failure.

**35. How would you validate someone else's volatility model?**
Read the documentation for intended use and assumptions. Re-implement or reconcile a
core piece independently (as I did with GARCH and HAR). Check the data lineage. Run
out-of-sample backtests with the right tests (Kupiec, Christoffersen, ES, Basel zones)
against simple benchmarks (EWMA, HS). Do stress and sensitivity analysis. Then write
findings with a severity and a monitoring plan.

**36. Which model would you put in production and why?**
I would pick from the models rated Satisfactory in my report, preferring the simplest
one that sits in the MCS for volatility and passes the tail tests. In practice that
means a GARCH-type model with fat tails or FHS, because it is transparent, cheap to
monitor and explainable to a regulator. I would keep HS or EWMA as a benchmark
challenger.

## F. Engineering

**37. How is the project structured, and how do you reproduce the results?**
A `src/` package with data, features, models, backtest, evaluation, reporting and api
subpackages. All parameters are in YAML. `make all` runs data → backtest → fit →
report. Dependencies are pinned, seeds fixed and reruns identical. The report
generator writes every number in the README and docs, and nothing is typed by hand.

**38. Tell me about the API.**
FastAPI with pydantic schemas. `/forecast/{asset}` returns next-day volatility, VaR
and ES for a chosen model; there are also `/health`, `/assets` and `/models`. Path and
query parameters are validated by regex, unknown assets or models return 404, and the
service returns 503 before forecasts exist. It never fits models at request time; it
serves a JSON file that a scheduled GitHub Action refreshes each weekday and hot-reloads
when the file changes. It runs in Docker with docker-compose.

**39. What tests did you write?**
No-lookahead perturbation tests for every model family; GARCH-from-scratch vs `arch`;
HAR/OLS vs statsmodels; Kupiec, Christoffersen, traffic-light and Z2 size and power on
synthetic data with known violation rates; QLIKE and FZ0 properties on known values;
API endpoint tests; and an end-to-end run on simulated GARCH
data that must recover the true parameters. CI runs lint, the whole suite and a Docker
build-and-smoke test on every push to main and every pull request.

**40. What was the hardest bug?**
The no-lookahead test failed for EGARCH: forecasts *before* the cut-off differed
between runs. It turned out not to be leakage. `arch`'s simulation forecast ignores the
`random_state` argument and uses an unseeded generator. I fixed it by passing an
explicitly seeded sampler. It taught me that a strict reproducibility test catches more
than the one bug it was written for.
