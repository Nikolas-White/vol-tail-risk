# Resume bullets

Generated from the latest real-data run by `make report`. Every number and every
comparative claim ("did not beat", "breached ... x as often") is computed from the
results, so the bullets stay true if the results change. Edit the wording freely,
but keep the numbers tied to `reports/RESULTS.md` section 10.

**Project title for the resume:** *Volatility & Tail-Risk Forecast Validation (Python, GARCH, XGBoost, FastAPI)*, with a link to the GitHub repo.

<!-- AUTO:resume_bullets -->
- Built a leakage-tested walk-forward backtest of 15 volatility and VaR/ES models on 7 assets over 19 years in Python; XGBoost did not beat GARCH-type models on volatility (statistically tied in a Model Confidence Set).
- Diagnosed why fat-tailed GARCH breached 99% VaR on equity indices and tested a pre-registered skewness fix: exceedances moved SPX 1.6%->1.2%, TSX 1.8%->1.0% (target 1%); hypothesis supported.
- Validated 99%/97.5% VaR and ES with Kupiec, Christoffersen, Basel traffic-light and Acerbi-Szekely tests: Conditional EVT/GPD (GJR-t filter) was most reliable (11/14 checks passed); RiskMetrics EWMA breached 99% VaR 3.7x as often as expected in COVID.
- Served daily volatility/VaR/ES forecasts via a Dockerized FastAPI service refreshed by GitHub Actions; documented results in a bank-style model validation report.
<!-- /AUTO:resume_bullets -->

### Facts behind the bullets
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

### Tailoring tips
- **Model validation / risk analyst roles:** lead with bullet 3, and mention "SR 11-7-style validation report" and the Basel traffic light.
- **Quant research roles:** lead with bullet 2, and mention the Diebold-Mariano / Model Confidence Set methodology.
- **Data science roles:** lead with bullet 1 (engineering, leakage-proof evaluation) and bullet 4.
- Be ready to defend every number. `docs/INTERVIEW_QA.md` covers the 40 most likely questions.
