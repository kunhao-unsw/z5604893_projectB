# AGENTS.md — z5604893 Project B working instructions

This repository is my FINS3645 Project B. Before changing code, read
`PROJECT_BRIEF.md`, `context/DATA_GUIDE.md`, and the relevant Week 7–9 lecture
notes I provide. The brief controls if instructions conflict.

## Project rules

- Work only inside my own `z5604893_projectB` folder.
- Preserve the provided `src/data_access.py` and never commit raw source data.
- Keep the deployed Streamlit app lightweight: it reads precomputed files under
  `results/` and does not run NLTK, download data, or backtest interactively.
- Reproduce all outputs with `python scripts/run_part_b.py`.
- Use exact mandatory filenames from the brief.

## Method checks

- Calculate equity and crypto returns separately before calendar alignment.
- For combined funds, left-merge the already-calculated crypto daily returns
  onto the equity trading calendar. Weekend-only crypto moves are intentionally
  excluded. Never compound weekend returns into Monday.
- Use 252-day annualisation for equity/combined funds and 365 for crypto-only.
- Walk-forward weights may use only observations strictly before rebalance.
- Lag news sentiment by at least one trading day before using it in a fund.
- Score individual headlines before ticker-day and equal-weight sector averages.
- Report no-news treatment and coverage explicitly.
- Enforce long-only weights summing to one and the stated 20% asset cap,
  including after any sentiment tilt.
- Never hide solver failures; save diagnostics and test any fallback.
- Keep negative or underperforming results and explain them honestly.

## Verification and AI transparency

- Add focused tests for calendar handling, annualisation, look-ahead, weight
  constraints and app artifact loading.
- Run tests, `scripts/check_handin.py`, and a Streamlit smoke test before hand-in.
- Log substantive AI prompts, errors found, corrections, evidence, and what I
  still need to review in `ai/`.
- Do not invent economic interpretation for submission. Flag draft wording and
  results that I must verify and rewrite in my own words.
