# Self-quiz: can you defend this project?

A resume project only helps if you can explain it without notes. Use this file to
find out whether you can. **Do not put the project on your resume until you can do
Levels 1 and 2 out loud, without looking.** Level 3 is what turns it into *your*
project: change the code, predict what happens, then run it and check.

Answers are hidden in the collapsible sections. Say your answer out loud first, then open them.

---

## Level 1: explain it (aim: 30 seconds per question, no notes)

1. What problem does the project answer, in one sentence?
<details><summary>Answer</summary>
When you forecast volatility and tail risk honestly out of sample, which models hold up?
And specifically, does modelling skewness fix the equity VaR failures?
</details>

2. Why can't you measure daily volatility directly, and what did you use instead?
<details><summary>Answer</summary>
True variance is unobservable. Without intraday data there is no realized variance. I
used squared returns (unbiased but very noisy) and range estimators (Parkinson,
Garman-Klass: less noisy but they miss the overnight gap, so I rescale them with a
past-only ratio).
</details>

3. What does "no lookahead" mean here, and how do you *test* it?
<details><summary>Answer</summary>
A forecast for day t+1 uses only data available at the close of day t, and parameters
are estimated on data before the forecast date. The test runs the whole engine twice,
the second time with all data after a cut-off scrambled. Every forecast up to the day
after the cut-off must be identical to 1e-10.
</details>

4. Write the GARCH(1,1) equation and say what α, β and α+β mean.
<details><summary>Answer</summary>
σ²ₜ₊₁ = ω + α εₜ² + β σ²ₜ. α is the reaction to today's surprise, β the memory, and α+β
the persistence (shock half-life = ln 0.5 / ln(α+β)). The long-run variance is
ω/(1−α−β).
</details>

5. What does GJR add, and why does it matter for equities?
<details><summary>Answer</summary>
A term γ·1[εₜ<0]·εₜ², so negative shocks raise volatility more: the leverage effect.
Equity volatility reacts more to falls than to rallies.
</details>

6. Why QLIKE rather than MSE?
<details><summary>Answer</summary>
With a noisy but unbiased proxy, both rank models correctly in expectation (Patton
2011). MSE, though, is dominated by a few crisis days, because variance errors get
squared. QLIKE works on the ratio proxy/forecast and penalises under-prediction more,
which is the error that hurts in risk.
</details>

7. What is the Model Confidence Set, and why use it instead of just ranking?
<details><summary>Answer</summary>
It is the set of models that contains the best one with 90% confidence. Being in it
means "not statistically worse than the best". With 10 models, a plain ranking
over-claims, because many of the differences are noise.
</details>

8. VaR vs ES in one sentence each. Why did Basel move to 97.5% ES?
<details><summary>Answer</summary>
VaR is the loss you exceed only α of the time; ES is the average loss when you do
exceed it. VaR ignores tail severity and is not subadditive. 97.5% ES is close to 99%
VaR under a normal distribution, so capital stayed comparable while becoming
tail-sensitive.
</details>

9. What do Kupiec and Christoffersen each test? Give a model that passes one and fails the other.
<details><summary>Answer</summary>
Kupiec tests the *number* of exceptions; Christoffersen tests whether they *cluster*.
Unconditional HS can have about 1% exceptions overall (passing Kupiec) with all of them
in crisis months (failing Christoffersen).
</details>

10. Where do the Basel traffic-light cut-offs (4/9) come from?
<details><summary>Answer</summary>
If the 99% VaR model is correct, exceptions over 250 days follow Binomial(250, 0.01).
Green ends where the cumulative probability passes 95% (P(≤4) = 0.892, P(≤5) = 0.959)
and red starts where it passes 99.99% (10 exceptions).
</details>

11. Explain filtered historical simulation.
<details><summary>Answer</summary>
Divide past returns by their GARCH volatility to get unit-free shocks. Tomorrow's VaR is
GARCH's volatility forecast times the empirical quantile of those shocks. GARCH supplies
"how volatile now", history supplies the shape of the tail.
</details>

12. What is EVT/GPD doing that HS cannot?
<details><summary>Answer</summary>
It extrapolates. Excesses over a high threshold are approximately Generalized Pareto
(Pickands–Balkema–de Haan), so you can estimate quantiles beyond the handful of observed
extreme points.
</details>

13. Why did Student-t GJR fail at 99% on equities while filtered HS didn't?
<details><summary>Answer</summary>
Same volatility forecast, different tail shape. The symmetric t cannot capture negative
skew (crashes larger than rallies), so its left-tail quantile is too small. FHS uses the
empirical residuals, which contain the skew. That observation motivated the skewed-t
research question. Know what the verdict was: README and RESULTS.md section 0.
</details>

14. What is the skewed-t's extra parameter, and what sign did you expect?
<details><summary>Answer</summary>
λ in (−1, 1), Hansen's skewness parameter. I expected λ < 0 for equity indices (a heavier
left tail).
</details>

15. Why was the hypothesis "pre-registered"? Why does that matter?
<details><summary>Answer</summary>
The decision rule (C1–C3) was written down before running the model on real data, so I
couldn't tune the criteria to make the result look good. It is the same reason the
model-rating rule was fixed in advance.
</details>

16. What did XGBoost do, and why isn't that surprising?
<details><summary>Answer</summary>
It was competitive but not statistically better than GARCH-type models (check the
current numbers in RESULTS.md). Daily volatility is a smooth, persistent process that
GARCH and HAR already capture, and the extra signal is small relative to proxy noise.
</details>

17. Name two data problems you found, and how.
<details><summary>Answer</summary>
(a) The bad-tick filter was deleting the genuine COVID crash days of 12–13 March 2020,
found by listing what it dropped; fixed with a 20-day adaptive scale. (b) One garbage
FX low print on 2012-01-27 (a 42–59% intraday wick) dominated the range proxy, found
because GK/r² was 1.8–3.2× for the FX pairs; fixed with a past-only wick filter.
</details>

18. How is the API built, and why doesn't it fit models per request?
<details><summary>Answer</summary>
FastAPI with pydantic schemas serves a JSON file of next-day forecasts, refreshed by a
scheduled GitHub Action. Fitting is a batch job; serving should be fast and
deterministic.
</details>

19. What are the biggest limitations?
<details><summary>Answer</summary>
Free daily data, so noisy proxies and no realized volatility; univariate models, so no
portfolio or correlation risk; a backtest, not live; stress windows chosen with
hindsight; many tests, so some rejections happen by chance.
</details>

20. If you had two more months, what would you do?
<details><summary>Answer</summary>
Intraday realized variance; a portfolio VaR (DCC-GARCH or FHS on portfolio P&L);
sensitivity to estimation window, EVT threshold and refit frequency; implied volatility
as a feature.
</details>

---

## Level 2: point to the code (open the file and explain the exact lines)

1. Where is the timing convention defined (forecast at origin o labelled with date o+1)? → `src/voltail/backtest/walkforward.py::run_family`
2. Where does the GARCH variance recursion run as a linear filter? Why is that equivalent to a loop? → `garch_scratch.py::garch_variance`
3. Where is the cross-asset feature lagged strictly by a day? Which pandas argument does it? → `features/ml_features.py::cross_features`
4. Where is the 5-day target made to stop at the training cut-off? → `models/ml.py` (`ml_target(v_train, h)`)
5. Where is the past-only scaling of the range proxy? → `features/proxies.py::expanding_scale` and `walkforward.py::realized_targets` (`.shift(1)`)
6. Where is the Kupiec LR computed? Derive it on paper for 250 days with 7 exceptions. → `evaluation/var_tests.py::kupiec_pof`
7. Where is the skewed-t ES integral, and why the substitution u = eˢ? → `models/distributions.py::skewt_es`
8. Where is the pre-registered decision rule implemented? → `reporting/research.py::verdict`
9. Where are model ratings decided? → `evaluation/ratings.py`
10. Which test would fail if you joined S&P 500 data on the *same* date? → `tests/test_no_lookahead.py::test_cross_asset_driver_strictly_lagged` and the xgb perturbation test

---

## Level 3: break it and predict (do these yourself; write your prediction first)

Run `make test` after each change, then `git checkout .` to undo.

1. In `ml_features.py`, change `allow_exact_matches=False` to `True`. *Predict which tests fail.* Then run `pytest tests/test_no_lookahead.py`.
2. In `walkforward.py::realized_targets`, delete `.shift(1)` on `scale`. Predict whether any test catches it, then find the test that does and explain what it asserts.
3. Set `ewma_lambda: 0.97` in `config/default.yaml` and run `make backtest report`. *Predict*: does EWMA's 99% exceedance rate go up or down in COVID? Why?
4. Set `walkforward.window: rolling` with `rolling_obs: 1000`. Predict how GARCH persistence and the VaR exceedances in 2022 change.
5. Set `evt.threshold_quantile: 0.95`. Does EVT VaR get more or less stable across refits? Why (bias–variance)?
6. Set `cleaning.spike_lookback: 250` and `spike_mad_mult: 12`, re-run `python scripts/download_data.py --from-csv`, and read `data/processed/data_quality.json`. Which real days get deleted? (You should rediscover DEVELOPMENT_LOG #10.)
7. Change the rating rule in `ratings.py` to require only Kupiec. How many models become "Satisfactory"? Why is that a worse rule?
8. Replace QLIKE by MSE for the ranking (in `evaluation/run.py`, rank by `mse`). Does the ranking change? Which assets drive it?

---

## Level 4: extend it yourself (pick one; write it without AI help)

These are deliberately left undone so that one part of the repo is unmistakably yours,
and the interview story becomes "I extended it by…":

- **Two-asset portfolio VaR:** a 60/40 SPX/UST10 portfolio. Compare (a) FHS on the portfolio return series with (b) summing the asset VaRs. Show the diversification benefit and how it disappears in 2022, when stocks and bonds fell together.
- **Skewed residuals inside FHS/EVT:** filter with `gjr_skewt` instead of `gjr_t` and see whether the semi-parametric tail models change.
- **Sensitivity table:** run the backtest with refit frequency M/Q/Y and window expanding/rolling; report how the ratings move. Validation teams love this.

**Make the git history yours, too:** commit in small steps as you work through this
quiz (one commit per exercise or extension). A single giant initial commit is the first
thing a sceptical interviewer notices.
