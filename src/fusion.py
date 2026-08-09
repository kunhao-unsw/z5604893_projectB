"""Sentiment fusion utilities for Project B.

This file keeps the sentiment-tilt logic separate from the pure portfolio
optimisation code. The tilt is intentionally simple and explainable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def apply_sector_sentiment_tilt(
    weights: pd.Series,
    asset_meta: pd.DataFrame,
    sector_signal: pd.Series,
    tilt_strength: float = 0.30,
    min_multiplier: float = 0.50,
    max_multiplier: float = 1.50,
    max_weight: float = 0.20,
) -> pd.Series:
    """Tilt equity weights using lagged sector sentiment.

    Parameters
    ----------
    weights:
        Base portfolio weights indexed by ticker.
    asset_meta:
        DataFrame with columns asset, sector.
    sector_signal:
        Series indexed by sector containing lagged sentiment values known at
        the rebalance date.
    tilt_strength:
        How strongly weights respond to relative sector sentiment.

    Returns
    -------
    pd.Series
        Tilted and re-normalised weights.
    """
    if weights.empty:
        return weights

    meta = asset_meta[["asset", "sector"]].drop_duplicates().set_index("asset")
    aligned = weights.to_frame("weight").join(meta, how="left")

    signal = sector_signal.dropna().copy()
    if signal.empty or signal.std(ddof=0) == 0:
        return weights / weights.sum()

    z = (signal - signal.mean()) / signal.std(ddof=0)

    multipliers = aligned["sector"].map(z).fillna(0.0)
    multipliers = 1.0 + tilt_strength * multipliers
    multipliers = multipliers.clip(lower=min_multiplier, upper=max_multiplier)

    tilted = aligned["weight"] * multipliers
    if tilted.sum() <= 0 or not np.isfinite(tilted.sum()):
        return weights / weights.sum()

    tilted = tilted / tilted.sum()

    # The tilt is a second portfolio-construction step, so it must preserve the
    # same concentration rule as the base optimiser. Re-normalising by itself
    # can otherwise push a previously capped asset above the 20% limit.
    max_weight = max(float(max_weight), 1.0 / len(tilted))
    for _ in range(100):
        over = tilted > max_weight + 1e-12
        if not over.any():
            break
        excess = float((tilted.loc[over] - max_weight).sum())
        tilted.loc[over] = max_weight
        under = ~over
        room = max_weight - tilted.loc[under]
        room_total = float(room.clip(lower=0.0).sum())
        if room_total <= 0:
            break
        tilted.loc[under] += excess * room.clip(lower=0.0) / room_total

    return tilted / tilted.sum()
