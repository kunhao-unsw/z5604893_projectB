"""Investor-facing analytics used by the precomputed Streamlit app."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def align_fund_returns_for_allocation(
    fund_returns: pd.DataFrame,
    chosen_funds: list[str],
) -> tuple[pd.DataFrame, int]:
    """Align selected fund returns on dates shared by every chosen fund.

    All-crypto selections retain their native seven-day calendar and 365-period
    annualisation. Any allocation containing an equity or combined fund uses
    the shared equity-date observations and 252-period annualisation, matching
    the Project Brief's calendar-alignment convention.
    """
    if not chosen_funds:
        raise ValueError("Choose at least one fund.")
    missing = sorted(set(chosen_funds).difference(fund_returns.columns))
    if missing:
        raise ValueError(f"Unknown funds: {missing}")

    selected = fund_returns[chosen_funds].sort_index().dropna(how="any")
    if selected.empty:
        raise ValueError("Selected funds have no overlapping return observations.")

    all_crypto_only = all(name.startswith("Crypto ") for name in chosen_funds)
    periods_per_year = 365 if all_crypto_only else 252
    return selected, periods_per_year


def apply_management_fee(
    gross_returns: pd.Series,
    annual_fee: float,
    periods_per_year: int,
) -> pd.Series:
    """Deduct a stated annual management fee proportionally each period."""
    if annual_fee < 0 or annual_fee >= 1:
        raise ValueError("Annual fee must be between 0 and 1.")
    periodic_fee = 1.0 - (1.0 - annual_fee) ** (1.0 / periods_per_year)
    return (1.0 + gross_returns) * (1.0 - periodic_fee) - 1.0


def portfolio_summary(
    returns: pd.Series,
    periods_per_year: int,
    rf_annual: float = 0.0,
) -> dict[str, float]:
    """Return fact-sheet metrics for a custom allocation."""
    r = returns.dropna()
    if r.empty:
        raise ValueError("No aligned allocation returns are available.")

    growth = (1.0 + r).cumprod()
    drawdown = growth / growth.cummax() - 1.0
    annual_return = growth.iloc[-1] ** (periods_per_year / len(r)) - 1.0
    annual_volatility = r.std() * math.sqrt(periods_per_year)
    rf_period = (1.0 + rf_annual) ** (1.0 / periods_per_year) - 1.0
    standard_deviation = r.std()
    sharpe = (
        (r.mean() - rf_period) / standard_deviation * math.sqrt(periods_per_year)
        if standard_deviation > 0
        else np.nan
    )
    downside = np.minimum(r - rf_period, 0.0)
    downside_deviation = math.sqrt(float(np.mean(np.square(downside))))
    sortino = (
        (r.mean() - rf_period) / downside_deviation * math.sqrt(periods_per_year)
        if downside_deviation > 0
        else np.nan
    )
    threshold = float(r.quantile(0.05))
    tail = r.loc[r <= threshold]

    return {
        "annual_return": float(annual_return),
        "annual_volatility": float(annual_volatility),
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "max_drawdown": float(drawdown.min()),
        "var_95": float(-threshold),
        "cvar_95": float(-tail.mean()),
        "total_return": float(growth.iloc[-1] - 1.0),
    }


def worst_rolling_return(returns: pd.Series, window: int) -> float:
    """Worst compounded return over a fixed rolling valuation window."""
    rolling = (1.0 + returns).rolling(window).apply(np.prod, raw=True) - 1.0
    return float(rolling.min())
