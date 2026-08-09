"""Regression checks for the precomputed app artifacts."""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd


ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_fund_return_artifact_is_chronological():
    returns = pd.read_csv(
        ROOT / "results" / "data" / "fund_returns.csv",
        parse_dates=["date"],
    ).set_index("date")

    assert returns.index.is_monotonic_increasing
    assert not returns.index.duplicated().any()
    assert returns.shape[1] == 14


def test_precomputed_weights_obey_constraints():
    weights = pd.read_csv(ROOT / "results" / "data" / "fund_weights.csv")
    sums = weights.groupby(["date", "fund"])["weight"].sum()

    assert np.allclose(sums, 1.0, atol=1e-10)
    assert weights["weight"].min() >= -1e-12
    assert weights["weight"].max() <= 0.20 + 1e-10


def test_solver_fallbacks_are_visible_and_unused_in_final_results():
    diagnostics = pd.read_csv(
        ROOT / "results" / "tables" / "optimizer_diagnostics.csv"
    )

    assert {"solver_success", "used_fallback", "solver_message"}.issubset(
        diagnostics.columns
    )
    assert diagnostics["used_fallback"].sum() == 0


def test_final_metrics_include_teacher_requested_downside_measures():
    metrics = pd.read_csv(ROOT / "results" / "tables" / "performance_metrics.csv")
    assert {"sortino_ratio", "var_95", "cvar_95"}.issubset(metrics.columns)
    assert {"Equity Risk Parity", "Crypto Risk Parity", "Combined Risk Parity"}.issubset(
        set(metrics["fund"])
    )
    assert "MarketPulse TailGuard" in set(metrics["fund"])


def test_tailguard_crypto_sleeve_is_capped_at_thirty_percent():
    weights = pd.read_csv(ROOT / "results" / "data" / "fund_weights.csv")
    tailguard = weights.loc[weights["fund"].eq("MarketPulse TailGuard")]
    crypto = (
        tailguard.assign(
            crypto_weight=tailguard["weight"].where(
                tailguard["asset_class"].eq("Crypto"),
                0.0,
            )
        )
        .groupby("date")["crypto_weight"]
        .sum()
    )
    assert (crypto <= 0.30 + 1e-10).all()


def test_manual_sentiment_validation_is_complete_and_reproducible():
    sample = pd.read_csv(
        ROOT / "results" / "tables" / "sentiment_manual_validation.csv"
    )
    metrics = pd.read_csv(
        ROOT / "results" / "tables" / "sentiment_manual_validation_metrics.csv"
    )
    class_metrics = pd.read_csv(
        ROOT
        / "results"
        / "tables"
        / "sentiment_manual_validation_class_metrics.csv"
    )

    assert len(sample) == 150
    assert sample["headline_id"].is_unique
    assert set(sample["manual_label"].str.strip().str.lower()) == {
        "negative",
        "neutral",
        "positive",
    }
    assert set(metrics["model"]) == {"Plain VADER", "Finance-extended VADER"}
    assert metrics["labelled_headlines"].eq(150).all()
    assert metrics[["accuracy", "macro_f1"]].apply(
        lambda column: column.between(0.0, 1.0).all()
    ).all()
    assert len(class_metrics) == 6
    assert set(class_metrics["label"]) == {"negative", "neutral", "positive"}
