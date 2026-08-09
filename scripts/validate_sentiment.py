"""Prepare or evaluate a blind manual validation sample for the sentiment model.

First run: creates results/tables/sentiment_manual_validation.csv with blank
manual_label cells.  The student reads each headline and enters positive,
neutral or negative without seeing model predictions.  Second run: scores the
completed sample and writes overall metrics, class metrics, a confusion-matrix
table, a scored audit file and a confusion-matrix figure.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.sentiment import score_individual_headlines


ROOT = pathlib.Path(__file__).resolve().parent.parent
HEADLINES = ROOT / "results" / "data" / "headline_panel_projectB.csv"
SAMPLE = ROOT / "results" / "tables" / "sentiment_manual_validation.csv"
METRICS = ROOT / "results" / "tables" / "sentiment_manual_validation_metrics.csv"
CLASS_METRICS = (
    ROOT / "results" / "tables" / "sentiment_manual_validation_class_metrics.csv"
)
CONFUSION = (
    ROOT / "results" / "tables" / "sentiment_manual_validation_confusion_matrix.csv"
)
SCORED = ROOT / "results" / "tables" / "sentiment_manual_validation_scored.csv"
FIGURE = ROOT / "results" / "figures" / "sentiment_manual_validation_confusion.png"
LABEL_ORDER = ["negative", "neutral", "positive"]
VALID_LABELS = {"negative", "neutral", "positive"}


def _prepare_sample(sample_size: int = 150) -> None:
    headlines = pd.read_csv(HEADLINES, parse_dates=["date"])
    headlines["year"] = headlines["date"].dt.year
    groups = list(headlines.groupby(["sector", "year"], sort=True))
    per_group = max(1, sample_size // len(groups))
    selected = [
        group.sample(min(per_group, len(group)), random_state=5604893)
        for _, group in groups
    ]
    sample = pd.concat(selected, ignore_index=True).drop_duplicates("headline_id")
    if len(sample) < sample_size:
        remaining = headlines.loc[~headlines["headline_id"].isin(sample["headline_id"])]
        sample = pd.concat(
            [
                sample,
                remaining.sample(sample_size - len(sample), random_state=5604893),
            ],
            ignore_index=True,
        )
    sample = sample.sample(frac=1.0, random_state=5604893).head(sample_size)
    sample = sample[
        ["headline_id", "date", "ticker", "sector", "headline_text"]
    ].copy()
    sample["manual_label"] = ""
    sample["student_notes"] = ""
    sample.to_csv(SAMPLE, index=False)
    print(f"Created blind sample: {SAMPLE}")
    print("Label every row positive, neutral or negative, then run this script again.")


def _classification_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    accuracy = float((actual == predicted).mean())
    class_rows = _class_metrics(actual, predicted)
    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean([row["f1"] for row in class_rows])),
    }


def _class_metrics(actual: pd.Series, predicted: pd.Series) -> list[dict[str, float]]:
    rows = []
    for label in LABEL_ORDER:
        true_positive = int(((actual == label) & (predicted == label)).sum())
        false_positive = int(((actual != label) & (predicted == label)).sum())
        false_negative = int(((actual == label) & (predicted != label)).sum())
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "label": label,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int((actual == label).sum()),
                "predicted_count": int((predicted == label).sum()),
            }
        )
    return rows


def _confusion_counts(actual: pd.Series, predicted: pd.Series) -> np.ndarray:
    return np.array(
        [
            [int(((actual == truth) & (predicted == estimate)).sum()) for estimate in LABEL_ORDER]
            for truth in LABEL_ORDER
        ]
    )


def _save_confusion_figure(matrices: list[tuple[str, np.ndarray]]) -> None:
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(matrices), figsize=(10, 4.3), sharex=True, sharey=True)
    if len(matrices) == 1:
        axes = [axes]
    maximum = max(int(matrix.max()) for _, matrix in matrices)
    for axis, (model, matrix) in zip(axes, matrices):
        axis.imshow(matrix, cmap="Blues", vmin=0, vmax=maximum)
        axis.set_title(model)
        axis.set_xticks(range(len(LABEL_ORDER)), LABEL_ORDER, rotation=25, ha="right")
        axis.set_yticks(range(len(LABEL_ORDER)), LABEL_ORDER)
        axis.set_xlabel("Predicted label")
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = int(matrix[row, column])
                axis.text(
                    column,
                    row,
                    str(value),
                    ha="center",
                    va="center",
                    color="white" if value > maximum / 2 else "black",
                )
    axes[0].set_ylabel("Manual label")
    fig.suptitle("Blind manual validation of headline sentiment (n=150)")
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.19, top=0.82, wspace=0.22)
    temporary = FIGURE.with_name(f"{FIGURE.stem}.writing{FIGURE.suffix}")
    fig.savefig(temporary, dpi=200, bbox_inches="tight")
    plt.close(fig)
    temporary.replace(FIGURE)


def _evaluate_sample() -> None:
    sample = pd.read_csv(SAMPLE)
    labels = sample["manual_label"].fillna("").str.strip().str.lower()
    invalid = sorted(set(labels).difference(VALID_LABELS))
    if invalid:
        raise ValueError(
            "Complete every manual_label with positive, neutral or negative. "
            f"Invalid/blank values include: {invalid[:3]}"
        )

    scored = score_individual_headlines(sample)
    rows = []
    class_rows = []
    confusion_rows = []
    matrices = []
    for model, prediction in [
        ("Plain VADER", scored["base_label"]),
        ("Finance-extended VADER", scored["finance_label"]),
    ]:
        row = {"model": model, "labelled_headlines": len(sample)}
        row.update(_classification_metrics(labels, prediction))
        rows.append(row)
        for class_row in _class_metrics(labels, prediction):
            class_rows.append({"model": model, **class_row})
        matrix = _confusion_counts(labels, prediction)
        matrices.append((model, matrix))
        for row_index, actual_label in enumerate(LABEL_ORDER):
            for column_index, predicted_label in enumerate(LABEL_ORDER):
                confusion_rows.append(
                    {
                        "model": model,
                        "actual_label": actual_label,
                        "predicted_label": predicted_label,
                        "count": int(matrix[row_index, column_index]),
                    }
                )
    pd.DataFrame(rows).to_csv(METRICS, index=False)
    pd.DataFrame(class_rows).to_csv(CLASS_METRICS, index=False)
    pd.DataFrame(confusion_rows).to_csv(CONFUSION, index=False)
    audit_columns = [
        "headline_id",
        "date",
        "ticker",
        "sector",
        "headline_text",
        "manual_label",
        "base_compound",
        "base_label",
        "compound",
        "finance_label",
    ]
    scored.loc[:, audit_columns].to_csv(SCORED, index=False)
    _save_confusion_figure(matrices)
    print(f"Saved validation metrics: {METRICS}")
    print(f"Saved class metrics: {CLASS_METRICS}")
    print(f"Saved confusion matrix: {CONFUSION}")
    print(f"Saved scored audit: {SCORED}")
    print(f"Saved confusion figure: {FIGURE}")


def main() -> None:
    if not SAMPLE.exists():
        _prepare_sample()
    else:
        _evaluate_sample()


if __name__ == "__main__":
    main()
