"""Station 3 - VADER sentiment model and sector sentiment index.

The app should read the CSV outputs generated here. It should not rerun VADER
during deployment.
"""
from __future__ import annotations

import pandas as pd


FINANCE_LEXICON = {
    # Transparent Week 8 finance-domain extension to plain VADER.
    "beat": 1.5,
    "beats": 1.5,
    "outperform": 2.0,
    "outperformed": 2.0,
    "upgrade": 2.0,
    "upgraded": 2.0,
    "bullish": 2.0,
    "profit": 1.5,
    "profitable": 1.7,
    "surge": 1.8,
    "surged": 1.8,
    "rebound": 1.5,
    "rebounded": 1.5,
    "miss": -1.5,
    "misses": -1.5,
    "underperform": -2.0,
    "underperformed": -2.0,
    "downgrade": -2.0,
    "downgraded": -2.0,
    "bearish": -2.0,
    "loss": -1.5,
    "losses": -1.5,
    "slump": -1.8,
    "slumped": -1.8,
    "default": -2.5,
    "bankruptcy": -3.0,
    "fraud": -3.0,
    "layoffs": -1.8,
}


def _get_vader_analyzer():
    """Load NLTK VADER, downloading the lexicon locally if needed."""
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()
    except LookupError:
        import nltk
        nltk.download("vader_lexicon", quiet=True)
        from nltk.sentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()


def _label_from_compound(x: float) -> str:
    if x >= 0.05:
        return "positive"
    if x <= -0.05:
        return "negative"
    return "neutral"


def score_headlines(panel: pd.DataFrame) -> pd.DataFrame:
    """Score each headline, then average scores within each ticker-day.

    Input must contain:
    date, ticker, sector, headline_text, headline_count.

    The output keeps both plain-VADER and finance-extended scores so the
    extension is measurable rather than asserted.
    """
    required = {"date", "ticker", "sector", "headline_text", "headline_count"}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"score_headlines missing columns: {sorted(missing)}")

    headline_scores = score_individual_headlines(panel)

    ticker_day = (
        headline_scores.groupby(["date", "ticker", "sector"], as_index=False)
        .agg(
            headline_count=("headline_text", "size"),
            base_compound=("base_compound", "mean"),
            neg=("neg", "mean"),
            neu=("neu", "mean"),
            pos=("pos", "mean"),
            compound=("compound", "mean"),
        )
        .sort_values(["date", "sector", "ticker"])
        .reset_index(drop=True)
    )
    ticker_day["sentiment_score_0_100"] = (ticker_day["compound"] + 1.0) * 50.0
    ticker_day["sentiment_label"] = ticker_day["compound"].apply(_label_from_compound)
    ticker_day["finance_lexicon_effect"] = (
        ticker_day["compound"] - ticker_day["base_compound"]
    )
    return ticker_day


def score_individual_headlines(panel: pd.DataFrame) -> pd.DataFrame:
    """Return plain and finance-extended VADER scores for each headline."""
    if "headline_text" not in panel.columns:
        raise ValueError("score_individual_headlines requires headline_text.")

    base_analyzer = _get_vader_analyzer()
    finance_analyzer = _get_vader_analyzer()
    finance_analyzer.lexicon.update(FINANCE_LEXICON)

    df = panel.copy()
    df["headline_text"] = df["headline_text"].fillna("").astype(str)
    base = df["headline_text"].apply(base_analyzer.polarity_scores).apply(pd.Series)
    finance = df["headline_text"].apply(finance_analyzer.polarity_scores).apply(pd.Series)
    base = base.add_prefix("base_")
    scored = pd.concat(
        [
            df.reset_index(drop=True),
            base.reset_index(drop=True),
            finance.reset_index(drop=True),
        ],
        axis=1,
    )
    scored["base_label"] = scored["base_compound"].apply(_label_from_compound)
    scored["finance_label"] = scored["compound"].apply(_label_from_compound)
    return scored


def finance_lexicon_audit_table() -> pd.DataFrame:
    """Expose every custom term and score for transparent student review."""
    return pd.DataFrame(
        [
            {
                "term": term,
                "vader_valence": valence,
                "intended_direction": "positive" if valence > 0 else "negative",
                "review_status": "Retained in submitted finance extension",
            }
            for term, valence in sorted(FINANCE_LEXICON.items())
        ]
    )


def sector_sentiment_index(
    scores: pd.DataFrame,
    universe: pd.DataFrame,
    equity_calendar: pd.Series | pd.Index | list,
    no_news: str = "neutral",
) -> pd.DataFrame:
    """Build an equal-weight daily sector sentiment index.

    The baseline treatment is no_news='neutral':
    every ticker-day with no headline receives compound = 0. This makes the
    index conservative and avoids overstating sentiment on days with thin news.

    The output is sector-day level and includes lagged sentiment to avoid
    look-ahead bias in later portfolio fusion.
    """
    if no_news != "neutral":
        raise ValueError("This baseline implementation currently supports no_news='neutral'.")

    required_scores = {
        "date",
        "ticker",
        "sector",
        "compound",
        "base_compound",
        "pos",
        "neu",
        "neg",
        "headline_count",
    }
    missing_scores = required_scores.difference(scores.columns)
    if missing_scores:
        raise ValueError(f"sector_sentiment_index scores missing columns: {sorted(missing_scores)}")

    required_universe = {"ticker", "sector"}
    missing_universe = required_universe.difference(universe.columns)
    if missing_universe:
        raise ValueError(f"sector_sentiment_index universe missing columns: {sorted(missing_universe)}")

    cal = pd.to_datetime(pd.Series(equity_calendar)).dt.normalize().dropna().drop_duplicates()
    cal = cal.sort_values().reset_index(drop=True)

    uni = universe[["ticker", "sector"]].drop_duplicates().copy()
    uni["ticker"] = uni["ticker"].astype(str).str.upper().str.strip()
    uni["sector"] = uni["sector"].astype(str).str.strip()

    # Complete date x ticker grid so no-news ticker-days are explicitly neutral.
    grid = (
        pd.MultiIndex.from_product([cal, uni["ticker"]], names=["date", "ticker"])
        .to_frame(index=False)
        .merge(uni, on="ticker", how="left")
    )

    sc = scores.copy()
    sc["date"] = pd.to_datetime(sc["date"]).dt.normalize()
    sc["ticker"] = sc["ticker"].astype(str).str.upper().str.strip()

    daily = grid.merge(
        sc[
            [
                "date",
                "ticker",
                "compound",
                "base_compound",
                "pos",
                "neu",
                "neg",
                "headline_count",
            ]
        ],
        on=["date", "ticker"],
        how="left",
    )

    daily["has_news"] = daily["headline_count"].notna()
    daily["headline_count"] = daily["headline_count"].fillna(0).astype(int)

    # Neutral baseline for no-news ticker-days.
    daily["compound"] = daily["compound"].fillna(0.0)
    daily["base_compound"] = daily["base_compound"].fillna(0.0)
    daily["pos"] = daily["pos"].fillna(0.0)
    daily["neg"] = daily["neg"].fillna(0.0)
    daily["neu"] = daily["neu"].fillna(1.0)

    sector = (
        daily.groupby(["date", "sector"], as_index=False)
        .agg(
            sentiment_compound=("compound", "mean"),
            base_sentiment_compound=("base_compound", "mean"),
            sentiment_pos=("pos", "mean"),
            sentiment_neu=("neu", "mean"),
            sentiment_neg=("neg", "mean"),
            total_headlines=("headline_count", "sum"),
            tickers_with_news=("has_news", "sum"),
            ticker_count=("ticker", "nunique"),
        )
        .sort_values(["sector", "date"])
        .reset_index(drop=True)
    )

    sector["coverage_rate"] = sector["tickers_with_news"] / sector["ticker_count"]
    sector["sentiment_score_0_100"] = (sector["sentiment_compound"] + 1.0) * 50.0
    sector["finance_lexicon_effect"] = (
        sector["sentiment_compound"] - sector["base_sentiment_compound"]
    )

    # Lag by one equity trading day within each sector before any later trading use.
    sector["sentiment_lag1"] = sector.groupby("sector")["sentiment_compound"].shift(1)
    sector["sentiment_score_lag1"] = sector.groupby("sector")[
        "sentiment_score_0_100"
    ].shift(1)

    # Full-sample z-score is descriptive only. The expanding z-score is causal:
    # its input is already lagged, so a date-t trade uses no later than t-1 news.
    sector["sentiment_z_full_sample"] = sector.groupby("sector")[
        "sentiment_compound"
    ].transform(
        lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) > 0 else 0.0
    )

    def _causal_expanding_z(x: pd.Series) -> pd.Series:
        mean = x.expanding(min_periods=20).mean()
        std = x.expanding(min_periods=20).std(ddof=0).replace(0.0, float("nan"))
        return (x - mean) / std

    sector["sentiment_signal_z"] = (
        sector.groupby("sector", group_keys=False)["sentiment_lag1"]
        .apply(_causal_expanding_z)
        .clip(-3.0, 3.0)
    )
    sector["sentiment_21d_avg"] = (
        sector.groupby("sector")["sentiment_compound"]
        .transform(lambda x: x.rolling(21, min_periods=5).mean())
    )
    sector["sentiment_score_21d_avg"] = (
        sector.groupby("sector")["sentiment_score_0_100"]
        .transform(lambda x: x.rolling(21, min_periods=5).mean())
    )

    return sector
