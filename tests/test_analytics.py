"""Tests for investor-facing calendar alignment and fee analytics."""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.analytics import (  # noqa: E402
    align_fund_returns_for_allocation,
    apply_management_fee,
    portfolio_summary,
)


def test_mixed_allocation_uses_shared_equity_dates():
    dates = pd.date_range("2023-01-06", "2023-01-09", freq="D")
    returns = pd.DataFrame(
        {
            "Equity Equal Weight": [0.01, np.nan, np.nan, 0.02],
            "Crypto Equal Weight": [0.01, 0.10, 0.10, 0.10],
        },
        index=dates,
    )
    aligned, periods = align_fund_returns_for_allocation(
        returns,
        ["Equity Equal Weight", "Crypto Equal Weight"],
    )

    assert periods == 252
    assert pd.Timestamp("2023-01-07") not in aligned.index
    assert list(aligned.index) == list(pd.to_datetime(["2023-01-06", "2023-01-09"]))
    assert np.isclose(aligned.loc["2023-01-09", "Crypto Equal Weight"], 0.10)


def test_crypto_only_allocation_retains_native_daily_calendar():
    dates = pd.date_range("2023-01-06", "2023-01-09", freq="D")
    returns = pd.DataFrame(
        {"Crypto Equal Weight": [0.01, 0.10, 0.10, 0.10]},
        index=dates,
    )
    aligned, periods = align_fund_returns_for_allocation(
        returns,
        ["Crypto Equal Weight"],
    )

    assert periods == 365
    assert list(aligned.index) == list(dates)


def test_management_fee_reduces_terminal_wealth():
    returns = pd.Series([0.01] * 40)
    net = apply_management_fee(returns, annual_fee=0.01, periods_per_year=252)
    assert (1.0 + net).prod() < (1.0 + returns).prod()


def test_custom_summary_contains_downside_metrics():
    returns = pd.Series([0.01, -0.02, 0.004, -0.006, 0.008, 0.003])
    summary = portfolio_summary(returns, periods_per_year=252)
    assert {"sortino_ratio", "var_95", "cvar_95", "max_drawdown"}.issubset(summary)
