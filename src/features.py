"""Station 2 - return features and text assembly.

Part B reuses the Part A data foundation, but this file also keeps the
return and headline-panel construction reproducible inside Project B.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _normalise_date(s: pd.Series) -> pd.Series:
    """Convert any timezone-aware timestamps to timezone-naive daily dates."""
    dt = pd.to_datetime(s, utc=True, errors="coerce")
    return dt.dt.tz_convert(None).dt.normalize()


def daily_returns(prices: pd.DataFrame, price_col: str = "adjClose") -> pd.DataFrame:
    """Compute simple daily returns per ticker using adjusted close.

    Returns a long panel with:
    date, ticker, simple_return, and sector if the input contains sector.
    """
    required = {"date", "ticker", price_col}
    missing = required.difference(prices.columns)
    if missing:
        raise ValueError(f"daily_returns missing columns: {sorted(missing)}")

    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.sort_values(["ticker", "date"])

    out = df[["date", "ticker", price_col] + (["sector"] if "sector" in df.columns else [])].copy()
    out["simple_return"] = out.groupby("ticker", sort=False)[price_col].pct_change()
    out = out.dropna(subset=["simple_return"]).reset_index(drop=True)

    keep_cols = ["date", "ticker"]
    if "sector" in out.columns:
        keep_cols.append("sector")
    keep_cols.append("simple_return")
    return out[keep_cols]


def assemble_headline_panel(
    headlines: pd.DataFrame,
    equity_calendar: pd.Series | pd.Index | list,
) -> pd.DataFrame:
    """Align individual headlines to ticker-days on the equity calendar.

    Mapping rule:
    - If a headline is published on an equity trading day, map it to that same day.
    - If it is published on a weekend/holiday, push it forward to the next equity trading day.
    - If no future trading day exists inside the sample, drop that headline.

    Each headline remains a separate row for scoring. Aggregation happens only
    after VADER has scored every headline, matching the Week 8–9 workflow:
    headline -> ticker-day mean -> equal-weight sector mean.
    """
    required = {"date", "ticker", "sector", "title"}
    missing = required.difference(headlines.columns)
    if missing:
        raise ValueError(f"assemble_headline_panel missing columns: {sorted(missing)}")

    news = headlines.copy()
    news["headline_date"] = _normalise_date(news["date"])
    news["ticker"] = news["ticker"].astype(str).str.upper().str.strip()
    news["sector"] = news["sector"].astype(str).str.strip()
    news["title"] = news["title"].astype(str).str.strip()

    # Remove exact duplicate headlines only. Different headlines for the same
    # ticker-date must stay because they are separate information events.
    news = news.drop_duplicates(subset=["ticker", "headline_date", "title"])

    cal = pd.to_datetime(pd.Series(equity_calendar)).dt.normalize().dropna().drop_duplicates()
    cal = cal.sort_values().reset_index(drop=True)
    cal_values = cal.to_numpy(dtype="datetime64[ns]")

    headline_dates = news["headline_date"].to_numpy(dtype="datetime64[ns]")
    idx = np.searchsorted(cal_values, headline_dates, side="left")

    mapped = np.full(
        len(news),
        np.datetime64("NaT", "ns"),
        dtype="datetime64[ns]",
    )
    valid = idx < len(cal_values)
    mapped[valid] = cal_values[idx[valid]]

    news["date"] = pd.to_datetime(mapped)
    news = news.dropna(subset=["date"])

    keep = ["date", "ticker", "sector", "title"]
    for optional in ["publisher", "url"]:
        if optional in news.columns:
            keep.append(optional)

    panel = news[keep].rename(columns={"title": "headline_text"}).copy()
    panel = panel.sort_values(["date", "sector", "ticker", "headline_text"]).reset_index(drop=True)
    panel.insert(0, "headline_id", range(1, len(panel) + 1))
    panel["headline_count"] = 1
    panel["n_words"] = panel["headline_text"].str.split().str.len()
    return panel
