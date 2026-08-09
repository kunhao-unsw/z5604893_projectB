"""Focused tests for Project B portfolio calendar and backtest rules."""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.fusion import apply_sector_sentiment_tilt  # noqa: E402
from src.portfolios import (  # noqa: E402
    build_return_panels,
    optimise_weights,
    performance_metrics,
    walk_forward_backtest,
)


def test_crypto_returns_are_computed_before_equity_calendar_alignment():
    equity_dates = pd.to_datetime(["2020-01-03", "2020-01-06", "2020-01-07"])
    equity = pd.DataFrame(
        {
            "date": equity_dates,
            "ticker": "AAA",
            "adjClose": [100.0, 101.0, 102.0],
            "sector": "Tech",
        }
    )

    crypto_dates = pd.date_range("2020-01-03", "2020-01-07", freq="D")
    crypto = pd.DataFrame(
        {
            "date": crypto_dates,
            "ticker": "BTC-USD",
            "adjClose": [100.0, 110.0, 121.0, 133.1, 146.41],
        }
    )

    panels, _ = build_return_panels(equity, crypto)

    # The Brief requires a left merge of already-computed crypto returns onto
    # the equity calendar. Saturday and Sunday are excluded; Monday retains
    # only the native Sunday-to-Monday crypto return.
    monday = pd.Timestamp("2020-01-06")
    assert np.isclose(panels["combined"].loc[monday, "BTC-USD"], 0.10)

    # The standalone crypto panel retains its full 365-day calendar.
    assert pd.Timestamp("2020-01-04") in panels["crypto"].index
    assert pd.Timestamp("2020-01-05") in panels["crypto"].index


def test_sharpe_uses_mean_daily_excess_return_not_cagr():
    dates = pd.bdate_range("2021-01-01", periods=6)
    returns = pd.DataFrame(
        {"Fund": [0.01, -0.005, 0.002, 0.004, -0.001, 0.003]},
        index=dates,
    )

    metrics = performance_metrics(returns, periods_per_year=252)
    expected = returns["Fund"].mean() / returns["Fund"].std() * math.sqrt(252)
    assert np.isclose(metrics.loc[0, "sharpe_ratio"], expected)


def test_first_rebalance_weights_do_not_use_future_returns():
    dates = pd.bdate_range("2020-01-01", periods=90)
    rng = np.random.default_rng(7)
    returns = pd.DataFrame(
        rng.normal(0.0005, 0.01, size=(90, 3)),
        index=dates,
        columns=["A", "B", "C"],
    )
    metadata = pd.DataFrame(
        {
            "asset": ["A", "B", "C"],
            "sector": ["S1", "S2", "S3"],
            "asset_class": "Equity",
        }
    )

    baseline = walk_forward_backtest(
        returns,
        metadata,
        "Test",
        ["A", "B", "C"],
        "min_vol",
        window=40,
        max_weight=0.80,
    )

    changed = returns.copy()
    changed.iloc[60:] = changed.iloc[60:] * 20
    mutated = walk_forward_backtest(
        changed,
        metadata,
        "Test",
        ["A", "B", "C"],
        "min_vol",
        window=40,
        max_weight=0.80,
    )

    first_date = baseline.weights["date"].min()
    first_baseline = (
        baseline.weights.loc[baseline.weights["date"].eq(first_date)]
        .set_index("asset")["weight"]
        .sort_index()
    )
    first_mutated = (
        mutated.weights.loc[mutated.weights["date"].eq(first_date)]
        .set_index("asset")["weight"]
        .sort_index()
    )

    pd.testing.assert_series_equal(first_baseline, first_mutated)


def test_sentiment_tilt_reapplies_weight_cap():
    weights = pd.Series(
        [0.20, 0.20, 0.20, 0.20, 0.10, 0.10],
        index=list("ABCDEF"),
    )
    metadata = pd.DataFrame(
        {
            "asset": list("ABCDEF"),
            "sector": ["Positive"] * 3 + ["Negative"] * 3,
        }
    )
    sector_signal = pd.Series({"Positive": 2.0, "Negative": -2.0})

    tilted = apply_sector_sentiment_tilt(
        weights,
        metadata,
        sector_signal,
        tilt_strength=0.30,
        max_weight=0.20,
    )

    assert np.isclose(tilted.sum(), 1.0)
    assert tilted.max() <= 0.20 + 1e-10


def test_optimizer_returns_visible_diagnostics():
    rng = np.random.default_rng(11)
    returns = pd.DataFrame(
        rng.normal(0.0004, 0.01, size=(300, 6)),
        columns=list("ABCDEF"),
    )

    weights, diagnostics = optimise_weights(
        returns,
        method="min_vol",
        return_diagnostics=True,
    )

    assert np.isclose(weights.sum(), 1.0)
    assert diagnostics["method"] == "min_vol"
    assert "solver_message" in diagnostics
    assert "used_fallback" in diagnostics


def test_risk_parity_obeys_long_only_budget_and_asset_cap():
    rng = np.random.default_rng(21)
    returns = pd.DataFrame(
        rng.normal(0.0004, [0.008, 0.010, 0.012, 0.014, 0.016, 0.018], size=(300, 6)),
        columns=list("ABCDEF"),
    )
    weights, diagnostics = optimise_weights(
        returns,
        method="risk_parity",
        max_weight=0.20,
        return_diagnostics=True,
    )

    assert np.isclose(weights.sum(), 1.0)
    assert weights.min() >= -1e-12
    assert weights.max() <= 0.20 + 1e-10
    assert diagnostics["method"] == "risk_parity"


def test_min_cvar_obeys_crypto_sleeve_cap():
    rng = np.random.default_rng(31)
    returns = pd.DataFrame(
        rng.normal(0.0003, 0.012, size=(260, 6)),
        columns=["E1", "E2", "E3", "E4", "C1", "C2"],
    )
    weights, diagnostics = optimise_weights(
        returns,
        method="min_cvar",
        max_weight=0.20,
        group_cap_assets=["C1", "C2"],
        group_cap=0.30,
        return_diagnostics=True,
    )

    assert np.isclose(weights.sum(), 1.0)
    assert weights.max() <= 0.20 + 1e-10
    assert weights[["C1", "C2"]].sum() <= 0.30 + 1e-10
    assert diagnostics["solver_success"]


def test_fact_sheet_metrics_include_downside_risk():
    dates = pd.bdate_range("2021-01-01", periods=8)
    returns = pd.DataFrame(
        {"Fund": [0.01, -0.02, 0.005, -0.01, 0.012, 0.003, -0.004, 0.006]},
        index=dates,
    )
    metrics = performance_metrics(returns, periods_per_year=252)

    assert {"sortino_ratio", "var_95", "cvar_95"}.issubset(metrics.columns)
    assert metrics.loc[0, "var_95"] > 0
    assert metrics.loc[0, "cvar_95"] >= metrics.loc[0, "var_95"]
