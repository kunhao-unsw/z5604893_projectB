# Final Project B results guide — verify and rewrite in your own words

This is an evidence map, not submission-ready report prose. The final report's
economic interpretation and recommendations must be the student's own writing.
Use only the regenerated CSVs in this folder; the earlier combined-fund numbers
are obsolete because they came from a weekend-compounding rule that conflicted
with the official Project Brief.

## Final fund design

- 14 core funds.
- Equity-only, crypto-only and combined universes each use equal weight,
  minimum volatility, tangency and risk parity (12 baseline funds).
- `MarketPulse TailGuard` is the innovation fund: combined universe, historical
  95% Minimum-CVaR, 20% asset cap and 30% aggregate crypto-sleeve cap.
- `Sentiment Tilt Equity` is the core fusion fund. Three extra tilt variants are
  kept in separate robustness artifacts rather than crowding the app fund menu.
- Monthly walk-forward rebalancing; weights use only observations strictly
  before each rebalance.
- Equity/combined: 252-observation window and annualisation. Crypto-only:
  365-observation window and annualisation.
- Long-only, fully invested, maximum individual asset weight 20%.
- Combined funds left-merge already-calculated crypto daily returns onto the
  equity trading calendar. Weekend-only crypto moves are intentionally excluded,
  as required by the Project Brief.

## Equations to typeset and number in the report

Define every symbol immediately below the relevant equation. Do not copy the
stochastic-process equations from the paper shown in class; that paper was an
equation-formatting example.

1. Equal weight: `w_i = 1/N`.
2. Minimum variance: `min_w w' Sigma_t w`.
3. Tangency: `max_w (w' mu_t - r_f) / sqrt(w' Sigma_t w)`.
4. Risk parity: minimise squared differences between each asset's variance
   contribution `w_i (Sigma_t w)_i` and the equal risk budget.
5. TailGuard: `min_{w, alpha} alpha + [1/((1-q)T)] sum max(-w'r_tau-alpha,0)`,
   where `q=0.95`.
6. Common constraints: `1'w=1` and `0 <= w_i <= 0.20`; TailGuard also has
   `sum_{i in crypto} w_i <= 0.30`.
7. OOS return: `r_{p,tau}=w_t' r_tau`, with `w_t` estimated only from dates
   before the rebalance.
8. Sentiment tilt: base weight multiplied by a clipped function of the lagged,
   expanding sector z-score, followed by renormalisation and cap reapplication.

State that `r_f=0` in this prototype. Annual return in the table is CAGR; Sharpe
uses arithmetic periodic mean excess return divided by periodic standard
deviation, multiplied by the square root of 252 or 365.

## Final numerical evidence

Source of truth: `results/tables/performance_metrics.csv`.

- Combined Tangency: 19.06% annual return, 23.41% volatility, Sharpe 0.86,
  Sortino 1.28, -23.14% maximum drawdown and 68.43% total return.
- Combined Risk Parity: 13.97% annual return, 16.10% volatility, Sharpe 0.89,
  Sortino 1.29, -19.98% maximum drawdown and 47.81% total return.
- Combined Equal Weight: 14.98% annual return, 21.25% volatility, Sharpe 0.76,
  Sortino 1.09, -28.75% maximum drawdown and 51.75% total return.
- Crypto Minimum Volatility: 58.40% annual return and 0.99 Sharpe, but 76.63%
  volatility and a -74.47% maximum drawdown.
- Equity Equal Weight: 12.64% annual return, 16.17% volatility, 0.82 Sharpe and
  -20.32% maximum drawdown.
- Equity Minimum Volatility: 12.60% volatility and -15.62% maximum drawdown, but
  only 5.79% annual return.

`cvar_95` is the positive magnitude of the average periodic loss among returns
at or below the fifth percentile. It is a one-observation tail-loss measure, not
an annual percentage.

## TailGuard result — retain the negative evidence

Compare `MarketPulse TailGuard` with Combined Minimum Volatility and Combined
Risk Parity rather than presenting it in isolation.

- TailGuard: 3.28% annual return, 13.72% volatility, 0.30 Sharpe, 0.42 Sortino,
  1.89% 95% daily CVaR loss, -18.37% maximum drawdown and 10.12% total return.
- Combined Minimum Volatility: 5.86% annual return, 12.61% volatility, 1.76%
  95% daily CVaR loss and -15.77% maximum drawdown.
- TailGuard's crypto sleeve never approached the 30% cap in the realised run;
  its maximum was about 5.86%.

The OOS result does not show that the CVaR method dominated the simpler
minimum-volatility fund. That is still valid innovation evidence because the
method was implemented, constrained, tested and evaluated honestly. Explain in
your own words why an objective estimated from a rolling historical window need
not deliver the best realised tail risk in a later period.

## Sentiment and fusion evidence

Headline workflow:

1. map weekend/holiday headlines to the next equity trading date;
2. preserve casing, punctuation, negation and intensifiers for VADER;
3. score each headline separately with plain and finance-extended VADER;
4. average to ticker-day, then equal-weight tickers within each sector;
5. fill no-news ticker-days with neutral score zero and report coverage;
6. display `(compound + 1) x 50` on a 0–100 scale;
7. lag one equity trading day and use an expanding causal z-score for trading.

The fixed sentiment tilt underperformed all four base methods in annual return
and Sharpe. From `fusion_robustness.csv`, annual-return differences (tilt minus
base) are approximately -1.73 percentage points for equal weight, -0.92 for
minimum volatility, -1.16 for tangency and -1.48 for risk parity. Do not hide
this. It is evidence that the simple headline timing rule was not robust in this
sample.

`sentiment_model_validation.csv` is a full-sample plain-versus-extended
diagnostic, not ground truth. The separate blind manual-validation exercise is
now complete. The student independently labelled 150 headlines before model
predictions were revealed: 68 positive, 59 neutral and 23 negative.

- Plain VADER: 46.0% accuracy and 0.412 macro-F1.
- Finance-extended VADER: 48.0% accuracy and 0.428 macro-F1.
- The extension changed 4 of 150 predictions; 3 changes corrected an error and
  none changed a previously correct prediction to an error.
- Finance-extended negative-class recall was only 21.7% (5 of 23), so the model
  still struggled with negative financial headlines. Do not overstate the
  modest overall improvement.

Evidence is stored in `sentiment_manual_validation_metrics.csv`,
`sentiment_manual_validation_class_metrics.csv`,
`sentiment_manual_validation_confusion_matrix.csv` and
`sentiment_manual_validation_scored.csv`. The visual exhibit is
`results/figures/sentiment_manual_validation_confusion.png`.

## Main exhibit paths

- Table 1 performance metrics:
  `results/tables/performance_metrics.csv`
- Figure — growth by three universes:
  `results/figures/fund_growth_1dollar.png`
- Figure — drawdown by three universes:
  `results/figures/fund_drawdowns.png`
- Figure — OOS risk-return map:
  `results/figures/fund_risk_return.png`
- Figure and table — latest-window efficient frontier (explicitly label this as
  an in-sample diagnostic, not OOS performance):
  `results/figures/efficient_frontier_latest_window.png` and
  `results/tables/efficient_frontier_latest_window.csv`
- Weight histories: `results/figures/weights_*.png`
- Combined crypto-sleeve comparison:
  `results/figures/weights_combined_methods_comparison.png`
- Sector sentiment small multiples:
  `results/figures/sector_sentiment_index.png`
- Required base-versus-tilt exhibit:
  `results/tables/fusion_before_after.csv` and
  `results/figures/fusion_before_after.png`
- Four-method fusion robustness:
  `results/tables/fusion_robustness.csv` and
  `results/figures/fusion_robustness.png`
- Lexicon audit: `results/tables/finance_lexicon_audit.csv`
- Blind manual sentiment-validation metrics and class diagnostics:
  `results/tables/sentiment_manual_validation_metrics.csv` and
  `results/tables/sentiment_manual_validation_class_metrics.csv`
- Blind manual sentiment-validation confusion matrices:
  `results/tables/sentiment_manual_validation_confusion_matrix.csv` and
  `results/figures/sentiment_manual_validation_confusion.png`
- Solver audit: `results/tables/optimizer_diagnostics.csv`

## Final quality evidence

- 24 automated tests pass, including a six-tab Streamlit AppTest render.
- 612 monthly rebalance records are stored across the core funds and fusion
  robustness variants.
- No solver failure or fallback is present.
- Every individual weight is at or below 20%.
- All 23 PNG artifacts pass image-integrity verification.

For every exhibit in the report, add a self-contained caption, units, sample
dates, method and evidence-based interpretation. Keep the main written narrative
within ten pages and move supporting exhibits to an appendix rather than
shrinking the font.
