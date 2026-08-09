# MarketPulse Funds — FINS3645 Project B

Student: z5604893

MarketPulse is a Streamlit investment-research prototype containing 14 core
walk-forward funds across equity-only, crypto-only and combined universes. The
fund menu includes equal weight, minimum volatility, tangency and risk parity,
plus a downside-focused Minimum-CVaR fund (MarketPulse TailGuard) and a lagged
sentiment-tilted equity fund. It also includes a finance-extended sector-news
index and a four-method sentiment-fusion robustness test.

## How to run

    pip install -r requirements.txt -r requirements-dev.txt   # dev adds nltk (VADER)
    python scripts/run_part_b.py
    pytest
    python scripts/validate_sentiment.py   # prepare/score the manual validation sample
    python scripts/check_handin.py
    streamlit run streamlit_app.py

Load raw data through src/data_access.py (see context/DATA_GUIDE.md); never commit
raw data. The deployed app, by contrast, reads your precomputed artifacts from
results/ - those ARE committed.

## Main design choices

- Returns are calculated inside each asset family's own calendar. For combined
  funds, the already-calculated crypto daily returns are left-merged onto the
  equity trading calendar, intentionally excluding weekend-only crypto moves as
  required by the Project Brief.
- Equity/combined results use 252-day annualisation; crypto-only uses 365.
- Monthly portfolio weights use only returns strictly before the rebalance date.
- Equal weight provides a benchmark for minimum-volatility, tangency and risk
  parity methods.
- MarketPulse TailGuard minimises historical 95% CVaR with a 20% asset cap and
  a 30% aggregate crypto-sleeve cap. Its result is retained even when it does
  not outperform simpler baselines.
- Every headline is scored before ticker-day and ticker-equal sector aggregation.
- The transparent finance lexicon is evaluated against plain VADER.
- The fusion signal is lagged, causally standardised, and capped after tilting.
- The same fixed sentiment tilt is evaluated across four base methods rather
  than selected after seeing which method performs best.
- Solver status, fallback use, data quality and fusion differences are saved as
  separate audit tables.
- Fact sheets include Sharpe, Sortino, 95% VaR, 95% CVaR and maximum drawdown.

## Repository structure

- streamlit_app.py    the app entrypoint (repo root)
- .streamlit/         app config
- PROJECT_BRIEF.md    the full assignment brief for your course (read this first)
- src/                your code (data_access is provided; portfolios/sentiment/fusion are yours)
- scripts/            runnable scripts that reproduce your results
- results/            your outputs: figures in results/figures/, tables in results/tables/, app data artifacts in results/data/
- context/            provided data guide and project context (do not edit)
- report/             your report - see report/OUTLINE.md (author in Word, submit report.pdf)
- ai/                 your prompt logs and AI notes
- requirements-dev.txt build/repro-only deps (nltk and pytest); keep them out of the deployed app
- AGENTS.md          project-specific Codex instructions
- ai/                prompt logs and review notes

## Deploy + hand in

This folder is its own GitHub repo, independent of fins-agent. Your AI agent can run
the check and push the repo; the browser deploy is yours (it needs your login). See
PROJECT_BRIEF.md Appendix D and docs/STUDENT_DEPLOY.md (in this folder). In short:

    python scripts/check_handin.py        # your agent can run this
    # commit your precomputed app artifacts under results/ (the app reads them)
    # git init in this folder, then push the contents to a NEW private GitHub repo

Then connect the repo on share.streamlit.io (entrypoint streamlit_app.py). At
hand-in, make the repo PUBLIC, submit the live URL + repo link, and also zip this
whole folder and upload the zip to Moodle.

## Manual work that cannot be fabricated

`results/tables/sentiment_manual_validation.csv` is a blind 150-headline sample.
The student independently entered `positive`, `neutral` or `negative` in every
`manual_label` cell before model predictions were revealed. Running
`python scripts/validate_sentiment.py` now reproduces the accuracy, macro-F1,
per-class diagnostics, scored audit table and confusion-matrix figure.

The economic interpretation, recommendations and final report wording must be
the student's own. `report/RESULTS_GUIDE.md` maps the final artifacts but is not
submission-ready report prose.
