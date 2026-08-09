# Submission checklist - Part B

Tick every item before you hand in. Run `python scripts/check_handin.py` to verify
the mechanical ones.

- [x] Folder is named z5604893_projectB.
- [ ] report/report.pdf is present (authored in Word, exported to PDF; max 10 pages
      of narrative - exhibits may go in an appendix).
- [ ] The report includes every required exhibit from PROJECT_BRIEF.md, Section 5
      ("Required exhibits (Part B)"), each captioned and interpreted.
- [ ] The report numbers have been replaced with the final regenerated CSV
      values, especially every combined-fund result.
- [ ] The report includes numbered project equations, defines every symbol and
      contains a complete reference list.
- [x] At least the required combined fund with two methods, backtested
      out-of-sample with no look-ahead, with a fact sheet.
- [x] streamlit_app.py passes an automated local smoke test.
- [ ] The GitHub repo is PUBLIC and the live Streamlit app loads.
- [x] Raw data loads through src/data_access.py; no raw data or secrets committed.
      (Your derived results/ artifacts - the CSVs the app reads - ARE committed.)
- [x] AGENTS.md contains the actual Codex working instructions.
- [x] ai/ contains the prompt logs and correction evidence.
- [x] I manually labelled all 150 rows in
      results/tables/sentiment_manual_validation.csv and reran
      scripts/validate_sentiment.py before claiming manual validation.
- [ ] The writing and interpretation are your own.
- [ ] Submit: the zip to Moodle, the public repo link, and the live Streamlit URL.
