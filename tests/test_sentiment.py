"""Focused tests for headline aggregation and look-ahead-safe sentiment."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.features import assemble_headline_panel  # noqa: E402
from src.sentiment import score_headlines, sector_sentiment_index  # noqa: E402


def test_weekend_headline_maps_to_monday_and_is_tradable_tuesday():
    calendar = pd.to_datetime(["2023-01-06", "2023-01-09", "2023-01-10"])
    news = pd.DataFrame(
        {
            "date": ["2023-01-07"],
            "ticker": ["AAA"],
            "sector": ["Tech"],
            "title": ["Company receives analyst upgrade"],
        }
    )
    panel = assemble_headline_panel(news, calendar)
    assert panel.loc[0, "date"] == pd.Timestamp("2023-01-09")

    scores = score_headlines(panel)
    universe = pd.DataFrame({"ticker": ["AAA"], "sector": ["Tech"]})
    index = sector_sentiment_index(scores, universe, calendar)
    monday = index[index["date"].eq(pd.Timestamp("2023-01-09"))].iloc[0]
    tuesday = index[index["date"].eq(pd.Timestamp("2023-01-10"))].iloc[0]
    assert np.isnan(monday["sentiment_lag1"]) or monday["sentiment_lag1"] == 0
    assert np.isclose(tuesday["sentiment_lag1"], monday["sentiment_compound"])


def test_headlines_are_scored_before_ticker_day_average():
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-03", "2023-01-03"]),
            "ticker": ["AAA", "AAA"],
            "sector": ["Tech", "Tech"],
            "headline_text": [
                "Company receives an upgrade",
                "Company receives a downgrade",
            ],
            "headline_count": [1, 1],
        }
    )
    scores = score_headlines(panel)
    assert len(scores) == 1
    assert scores.loc[0, "headline_count"] == 2
    assert {"base_compound", "compound", "finance_lexicon_effect"}.issubset(
        scores.columns
    )


def test_expanding_signal_does_not_change_when_future_sentiment_changes():
    calendar = pd.bdate_range("2023-01-02", periods=60)
    universe = pd.DataFrame({"ticker": ["AAA"], "sector": ["Tech"]})
    base_scores = pd.DataFrame(
        {
            "date": calendar,
            "ticker": "AAA",
            "sector": "Tech",
            "compound": np.linspace(-0.2, 0.2, len(calendar)),
            "base_compound": np.linspace(-0.2, 0.2, len(calendar)),
            "pos": 0.2,
            "neu": 0.6,
            "neg": 0.2,
            "headline_count": 1,
        }
    )
    changed_scores = base_scores.copy()
    changed_scores.loc[changed_scores["date"] > calendar[39], "compound"] = -0.9

    first = sector_sentiment_index(base_scores, universe, calendar)
    second = sector_sentiment_index(changed_scores, universe, calendar)
    cutoff = calendar[39]
    left = first.loc[first["date"] <= cutoff, "sentiment_signal_z"].reset_index(drop=True)
    right = second.loc[second["date"] <= cutoff, "sentiment_signal_z"].reset_index(drop=True)
    pd.testing.assert_series_equal(left, right)
