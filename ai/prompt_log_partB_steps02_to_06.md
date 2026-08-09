# Part B Prompt Log 02–06 — Funds, Sentiment, App and Final Audit

> **Historical checkpoint.** This file documents Steps 02–06 exclusively. The 10 funds,
> 13 passing tests, 360 rebalance records and five-tab app below were
> correct at that checkpoint but were updated by Steps 07–09. The final Step
> 09 verification found 14 funds, 24 passing tests, 612
> optimiser-diagnostic rows, six tabs and the Project Brief's left-merge
> calendar rule.

## 1. Full revision request

### What I wanted

Following the initial Codex-assisted review, I requested Codex to complete the revised Project B as a
working folder I could open and inspect, not just tell me to install every file.
I wanted it to write the outputs directly instead of assuming I'd know how to
run all the files myself.

### Prompt summary

> Please write the revised code and finish the working folder for me, because
> I don't know how to install each file again myself.

Following the initial Codex-assisted review, I requested Codex to complete the revised Project B as a
working folder I could open and inspect.

I asked it to write the revised code and outputs directly, because I didn't
want only general installation instructions.

I repeatedly asked Codex to follow the Project Brief and the Week 7–9 lecture
material, but not to add complicated features only to try to look advanced.

### Problems found in my first version

While performing my own output-level audit, I saw that the first
`performance_metrics.csv` contained only five funds, with no crypto-only funds
or equal-weight benchmarks. This made the investor comparison too narrow.

I also saw when comparing the results folder with the Project Brief that there was
no dedicated base-versus-sentiment-fusion comparison table and figure.

When checking the sentiment outputs against the Week 9 lecture, I saw that
the first version reported VADER compound scores mostly, not the
0–100 sentiment scale used in the lecture.

Codex's technical code review also identified deeper implementation issues:
optimisation failures silently returned equal weights, headlines within the same
ticker-day were joined together and then scored only once, and the original
version used plain VADER only, not testing a transparent finance-domain extension.
The sentiment tilt normalised weights but didn't reapply the 20% asset cap, the
allocation tool converted missing fund returns to zero, and the allocation tool
also calculated Sharpe as CAGR divided by volatility.

I personally identified the output-level omissions shown above by looking at
the CSVs, results folder and lecture requirements. I didn't personally discover
every technical item in the deeper code-review list.

### What Codex produced

1. 10 out-of-sample funds: equity-only, crypto-only and combined universes,
   each with equal-weight, minimum-volatility and tangency methods plus the
   sentiment-tilted equity fund.
2. A 252-observation estimation and annualisation convention for equity and
   combined funds and 365 for crypto-only funds.
3. Solver diagnostics for each rebalance, including whether it succeeded, its
   message, how many iterations, whether it fell back, the objective value and the
   maximum weight.
4. Individual-headline VADER scoring, then ticker-day means and
   ticker-equal sector means.
5. Plain VADER and a small transparent finance-extended VADER so the
   effect of the extra lexicon can be measured.
6. Week 9 0–100 sentiment scale, descriptive full-sample z-score and
   causal expanding z-score. Only the one-day-lagged causal signal is used for
   the trading tilt.
7. Second 20% cap enforcement after sentiment tilt.
8. Fusion before/after table and figure.
9. Revised Streamlit investor journey and allocation tool that doesn't
   convert missing observations to zero but uses only common return dates.
10. Replacement `AGENTS.md` with instructions actually used for
    this project.

### Risk I asked the audit to control

My primary fear was that the revised project should follow the brief rather
than add complicated methods only to seem advanced. The finance lexicon
was therefore small and visible in `src/sentiment.py`. The deployed
app reads precomputed results and doesn't download data, run VADER or rerun the
backtest during interaction.

## 2. Result that I must not hide

What the final data showed was that the sentiment-tilted equity fund underperformed
the base equity tangency fund. Its annualised return was about 6.10% versus 7.26%,
and its Sharpe ratio was about 0.42 compared to 0.49. Codex kept that negative result
instead of trying to change the model until it looked successful. I have to write
the economic interpretation of this result in my own words in the report.

## 3. Dark Mode problem that I personally found

### Prompt summary

When I opened the revised Streamlit app in Dark Mode, I found that most of
the text was almost invisible. I told Codex that the app wasn't readable
because the text had almost no contrast with the page background.

> I found a problem when I opened the app in Dark Mode, and most of the text
> became almost invisible.

### What I checked

This was a problem I personally found by using the app, not a problem first
identified by Codex. The calculations still loaded, but the text and page
background had almost no contrast.

### What Codex changed

Codex identified that the custom CSS had forced a light background while Streamlit
continued to use light text from its Dark theme. It replaced the fixed colours
with Streamlit's `--background-color`, `--secondary-background-color` and
`--text-color` theme variables. I then received a revised version that was designed to
stay readable in both Light and Dark modes.

## 4. Final rule-by-rule audit

### Prompt summary

I asked Codex to audit the full ZIP once more, to check that the code followed
the Project Brief and the Week 7–9 lecture material before submission.

### An error in the earlier AI-assisted revision

The first AI-assisted revision wasn't automatically correct. During the later
final ZIP audit, Codex found that concatenating crypto fund returns with
equity-calendar returns using `sort=False` had left the combined CSV date index
non-monotonic. Total return, volatility and Sharpe were all unchanged, but the
paths of crypto fund growth and their maximum drawdowns depended on the wrong
row order.

I didn't find this chronological-order bug myself. I requested the additional
final audit and the later audit caught an error that had been missed in the
earlier AI-assisted revision. Codex added `sort_index()` after the concatenation,
rebuilt all the results, and added an artifact-level chronological-order test. This
was a good example of why I didn't treat the first AI output as automatically
correct.

The audit also revealed that a ZIP created in Finder included `.DS_Store` and
`__MACOSX` metadata. Codex created a clean archive that excludes these files.

## 5. Step 06 checkpoint evidence reported by Codex

- The full build completed using official hosted data.
- 10 tests passed before the final ZIP audit. Then three artifact-level tests
  were added, so the Step 06 checkpoint suite included 13 passing tests.
- The future-data mutation tests passed for both portfolio weights and the
  expanding sentiment signal.
- Weekend news maps forward to Monday and first affects the lagged signal on
  Tuesday.
- The Step 06 artifacts include 360 monthly fund rebalances and no solver
  fallbacks.
- Maximum post-tilt individual asset weight is 20%.
- Final `fund_returns.csv` date column is chronological and has no duplicate
  dates.
- Automated Streamlit smoke test reported no exceptions across five tabs.
- `scripts/check_handin.py` passed all code-folder checks; the report remains to
  be written and added separately.

These tests and numerical checks were run by Codex in its own workspace. I
have opened and used the app myself, including identifying the Dark Mode
problem, but I'll need to run the final commands myself before I submit.

## 6. Work that remains my responsibility

- Look at the finance lexicon values in `src/sentiment.py` and keep only the terms I
  understand and can justify.
- Run `python -m pytest -q`, `python scripts/check_handin.py` and
  `streamlit run streamlit_app.py` locally.
- Confirm fund names and wording shown to users.
- Cross-check every number to use in my report against the final CSV files.
- Write economic interpretation, recommendations, limitations and
  citations in my own words.
- Finish GitHub and Streamlit Community Cloud deployment using my own
  accounts.
