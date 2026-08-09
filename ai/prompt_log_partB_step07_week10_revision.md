# Prompt Log — Part B Step 07: Week 10 and Teacher-Feedback Revision

## My prompt

> Based on the analysis above, please generate the best version of the code.
> It must follow the classroom rules and all of the modification
> recommendations described above.

## Superseded calendar conclusion

This log preserves the outcome of Step 07, including an AI mistake. Step 09 records that a later AI-assisted review raised the weekend-compounding issue, after which I checked the official Brief, confirmed that the 33.1% treatment conflicted with the required left merge, and asked Codex to reverse it. The compliant Monday value in the synthetic example is 10%, not 33.1%.

## Actions of Codex

Codex created a separate working copy and did not overwrite my uploaded ZIP. It then updated the portfolio, analytics, build, test and Streamlit files. The main updates included:

- combined-fund calendar processing revised to compound all native crypto returns between equity valuation dates—an AI-assisted change later found to conflict with the Project Brief and reversed in Step 09;
- equity, crypto and combined risk-parity funds added;
- MarketPulse TailGuard, a combined historical Minimum-CVaR fund with a 20% individual asset cap and 30% crypto-sleeve cap;
- Sortino, 95% VaR and 95% CVaR added to fund and allocation analytics;
- one fixed sentiment tilt tested across equal weight, minimum volatility, tangency and risk parity instead of only reporting the tangency comparison;
- latest-window efficient-frontier diagnostic was added and crowded charts redesigned using panels, percentage axes, colour and marker legends;
- allocation tool changed to keep weekend crypto returns in mixed equity–crypto selections—also later aligned to the Brief in Step 09—and starting-amount and management-fee illustrations added;
- blind manual sentiment-validation sample created and evaluation script added;
- all artifacts rebuilt from the official hosted data and tests and hand-in checks run.

## Notable Step 07 AI mistake, corrected in Step 09

An earlier AI-assisted update specified that crypto returns were calculated before calendar alignment, but its combined panel limited the selection to only the crypto return shown on each equity date. Given three consecutive 10% crypto returns over Friday to Monday, the earlier test would expect 10% on Monday. At Step 07, Codex had mistakenly concluded that the assignment required a 33.1% compounded return and updated the regression test to reflect that specification. I subsequently compared this result with the official Brief and determined that the Brief clearly instructed a left-merge that retains Monday's native 10% return and omits weekend-only moves. Step 09 documents the compliant reversion.

The incorrect Step 07 rebuild had significantly altered the combined-fund outputs. For instance, Combined Tangency total return updated from approximately 68.43% to approximately 93.95%. These are historical checkpoint numbers: 93.95% is overridden and must not be used in the report. The final Brief-compliant value recorded in Step 09 is approximately 68.43%.

## Evidence verified in this update

- The full build reloaded the official price and news data and all tables, figures and app data were regenerated.
- At this Step 07 checkpoint, 22 automated tests passed, including an AppTest render of all six tabs.
- 14 core funds were generated, with 612 monthly rebalances recorded across core and sentiment-robustness backtests.
- No solver failure or fallback was detected in the final diagnostics.
- All individual weights were at or below 20%.
- The TailGuard crypto sleeve was held beneath its 30% cap.
- All 22 PNG exhibits passed an image-integrity test following detection and atomic regeneration of three incomplete PNG writes.

## Remaining actions my responsibility

At Step 07, it was still my responsibility to inspect the app on my local computer, complete the blind 150-headline labels manually before declaring manual validation, determine whether I accepted and understood the TailGuard design, and write the economic interpretation and recommendations in my own words. Step 08 documents the completion of manual labels, while Step 09 records the artifacts for the final report. I still must export `report/report.pdf`, deploy the app and submit the public repository and live URL.
