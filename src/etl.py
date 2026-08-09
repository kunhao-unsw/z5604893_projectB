"""Reusable Station 1 cleaning for the Project B build."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import data_access


PROJECT_END = pd.Timestamp("2023-12-31")


def _clean_prices(df: pd.DataFrame, asset_class: str) -> pd.DataFrame:
    required = {"date", "ticker", "adjClose"}
    if asset_class == "Equity":
        required.add("sector")
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{asset_class} prices missing columns: {sorted(missing)}")

    out = df.copy()
    out["date"] = (
        pd.to_datetime(out["date"], errors="coerce", utc=True)
        .dt.tz_convert(None)
        .dt.normalize()
    )
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    out["adjClose"] = pd.to_numeric(out["adjClose"], errors="coerce")
    out = out.loc[
        out["date"].notna()
        & (out["date"] <= PROJECT_END)
        & out["ticker"].ne("")
        & out["adjClose"].gt(0)
    ].copy()
    out = out.drop_duplicates(["ticker", "date"], keep="last")
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)


def load_clean_equities() -> pd.DataFrame:
    """Load and clean the 252-day equity panel."""
    return _clean_prices(data_access.load_equity_prices(), "Equity")


def load_clean_crypto() -> pd.DataFrame:
    """Load and clean the 365-day crypto panel, excluding 2024 rows."""
    return _clean_prices(data_access.load_crypto_prices(), "Crypto")


def load_clean_news() -> pd.DataFrame:
    """Load headlines, preserve VADER-relevant text, and remove exact duplicates."""
    required = {"date", "ticker", "sector", "title"}
    news = data_access.load_news_headlines().copy()
    missing = required.difference(news.columns)
    if missing:
        raise ValueError(f"News data missing columns: {sorted(missing)}")

    news["date"] = (
        pd.to_datetime(news["date"], errors="coerce", utc=True)
        .dt.tz_convert(None)
        .dt.normalize()
    )
    news["ticker"] = news["ticker"].astype(str).str.upper().str.strip()
    news["sector"] = news["sector"].astype(str).str.strip()
    news["title"] = news["title"].astype(str).str.strip()
    news = news.loc[
        news["date"].notna()
        & (news["date"] <= PROJECT_END)
        & news["ticker"].ne("")
        & news["title"].ne("")
    ].copy()
    news = news.drop_duplicates(["ticker", "date", "title"], keep="first")
    return news.sort_values(["date", "ticker", "title"]).reset_index(drop=True)


def integrity_audit(
    equities: pd.DataFrame,
    crypto: pd.DataFrame,
    news: pd.DataFrame,
) -> pd.DataFrame:
    """Quantify key post-cleaning integrity checks for the reproducibility pack."""
    rows = []
    for name, df in [
        ("equity_prices", equities),
        ("crypto_prices", crypto),
        ("news_headlines", news),
    ]:
        duplicate_keys = ["ticker", "date", "title"] if name == "news_headlines" else ["ticker", "date"]
        row = {
            "dataset": name,
            "rows": int(len(df)),
            "start_date": df["date"].min().date(),
            "end_date": df["date"].max().date(),
            "tickers": int(df["ticker"].nunique()),
            "duplicate_key_rows": int(df.duplicated(duplicate_keys).sum()),
            "missing_required_values": int(
                df[duplicate_keys].isna().to_numpy().sum()
            ),
        }
        if "adjClose" in df:
            returns = df.groupby("ticker")["adjClose"].pct_change(fill_method=None)
            row["nonpositive_prices"] = int((df["adjClose"] <= 0).sum())
            row["extreme_abs_return_gt_50pct"] = int(
                (returns.abs() > 0.50).sum()
            )
            row["nonfinite_returns"] = int(
                (~np.isfinite(returns.dropna().to_numpy())).sum()
            )
        rows.append(row)
    return pd.DataFrame(rows)
