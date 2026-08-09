# Prompt Log - Part B Step 09: Project Brief Calendar Correction

## Issue raised and my verification

After a later AI-assisted review raised the calendar issue, I checked the final code, tests, README and Streamlit Method Notes against the official Project Brief. I confirmed that `compound_returns_to_calendar()` had compounded Saturday, Sunday and Monday moves in the crypto returns into the Monday combined-fund return; but the Brief had stated that already-calculated crypto daily returns should be left-merged onto the equity trading calendar and that the combined fund should explicitly not include weekend-only gains and losses.

Representative prompt I sent after the issue was raised in an AI-assisted review:

> The first problem is very important: the current crypto-calendar treatment conflicts with the Project Brief in the ZIP. The Brief requires already-calculated crypto returns to be left-merged onto the equity calendar and intentionally excludes weekend-only crypto moves. The assignment rule must be followed.


## Earlier AI error

At Step 07, Codex had misunderstood holding-period compounding as the appropriate assignment rule. It had transformed the synthetic Friday-to-Monday test by altering the return from a 10% Monday crypto return to a 33.1% compounded return; it had also called this transformation a correction. While this compounding might be economically justifiable for a continuously held position, it did not follow the explicit data-construction rule in this Project Brief.

## Corrective implementation

Codex retained the native equity and crypto return computations but adjusted the combined panel to reindex already-calculated crypto daily returns onto the equity return dates. In the synthetic example in which the crypto had 10% returns on Saturday, Sunday and Monday, the combined panel now retains only Monday's native 10%. The standalone crypto panel still retains its seven-day calendar.

The allocation lab was also consolidated consistently: all-crypto selections still retain the seven-day calendar, while any selection with an equity or combined fund uses shared equity-date data. README, Streamlit Method Notes, tests, and the report evidence guide were adjusted. Every result artifact was reconstructed using the official data.

## Rebuild and verification evidence

The Brief-compliant reconstruction removed the artificial increase caused by the weekend-compounding implementation. In the rebuilt `results/tables/performance_metrics.csv`, Combined Tangency has an annual return of approximately 19.06%, annual volatility of 23.41%, a Sharpe ratio of 0.86, maximum drawdown of -23.14% and a total return of 68.43%. The superseded weekend-compounding artifact had reported approximately 24.82% annual return and 93.95% total return, so those superseded values must not appear in the report.

The final verification returned 14 core funds and 612 optimiser-diagnostic rows, with no solver failure or fallback. All 24 automated tests passed, including calendar regression tests and Streamlit AppTest. All 23 PNG exhibits passed the image-integrity check. My 150 completed manual sentiment labels were retained; the regenerated validation outputs report 48.0% accuracy and a macro-F1 of approximately 0.428 for finance-extended VADER.

## My remaining responsibilities

I must run the app locally, verify the revised values for the combined fund against `results/tables/performance_metrics.csv`, and use only the rebuilt CSVs and figures in the report. The economic interpretation of the report remains my own writing.
