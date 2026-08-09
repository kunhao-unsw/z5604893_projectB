# Part B Prompt Log — Step 08: Blind Manual Sentiment Validation

## Prompt summary

I had finished the 150 manual labels and asked Codex to check the completed
file and evaluate the sentiment models without changing my manual labels.

## Student contribution and authorship

I independently labelled the 150-headline blind sample in
`results/tables/sentiment_manual_validation.csv`. Allowed labels in the
dataset were positive, neutral and negative. AI did not provide or modify my 150 manual
labels. I finished with 68 positive, 59 neutral and 23 negative
labels.

Codex confirmed that all 150 labels were present and valid, headline IDs were unique and
that headline IDs, tickers, sectors, headline text and row order
agreed with the original blind sample. Excel had shown dates with slashes rather
than hyphens, but parsing and normalisation revealed that no dates had
actually changed.

Codex then executed `python scripts/validate_sentiment.py`. It modified the
validation script to generate reproducible overall metrics, per-class metrics,
confusion matrices, a scored audit table and a figure. It did not alter the
manual labels or retune the sentiment lexicon upon seeing the
validation results.

## Findings and critical review

Plain VADER: 46.0% accuracy; 0.412 macro-F1.

Finance-extended VADER: 48.0% accuracy; 0.428 macro-F1.

The finance extension altered 4 of 150 predictions. Three fixed errors and none
changed a previously correct prediction to wrong.

Finance-extended VADER correctly classified only 5 of 23 manually labelled negative
headlines (21.7% negative recall).

Manual evidence therefore supports only a modest improvement from the finance
extension. It does not support a claim of high headline-level sentiment accuracy.
The weak recall for the negative class is material to report. The economic
interpretation and final report wording are still my own
responsibility.
