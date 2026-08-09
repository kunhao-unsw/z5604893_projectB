"""Out-of-sample portfolio construction for Project B.

The functions here build systematic funds using walk-forward optimisation.
Weights at each rebalance date are estimated using only past returns.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.fusion import apply_sector_sentiment_tilt


EQUITY_TRADING_DAYS = 252
CRYPTO_TRADING_DAYS = 365
PROJECT_END = pd.Timestamp("2023-12-31")

# Backward-compatible name for the equity-calendar funds.
TRADING_DAYS = EQUITY_TRADING_DAYS


@dataclass
class BacktestResult:
    returns: pd.Series
    weights: pd.DataFrame
    diagnostics: pd.DataFrame


def _clean_price_panel(
    prices: pd.DataFrame,
    price_col: str,
) -> pd.DataFrame:
    """Normalise a price panel before calculating within-ticker returns."""
    required = {"date", "ticker", price_col}
    missing = required.difference(prices.columns)
    if missing:
        raise ValueError(f"Price panel missing columns: {sorted(missing)}")

    out = prices.copy()
    out["date"] = (
        pd.to_datetime(out["date"], errors="coerce", utc=True)
        .dt.tz_convert(None)
        .dt.normalize()
    )
    out = out.loc[out["date"].notna() & (out["date"] <= PROJECT_END)].copy()
    out["ticker"] = out["ticker"].astype(str).str.strip()
    out[price_col] = pd.to_numeric(out[price_col], errors="coerce")
    out = out.dropna(subset=["ticker", "date", price_col])
    out = out.drop_duplicates(["ticker", "date"], keep="last")
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)


def _wide_returns(prices: pd.DataFrame, price_col: str) -> pd.DataFrame:
    """Calculate simple returns inside an asset family's own calendar."""
    wide_prices = (
        prices.pivot(index="date", columns="ticker", values=price_col)
        .sort_index()
    )
    return wide_prices.pct_change(fill_method=None).dropna(how="all")


def build_return_panels(
    equity_prices: pd.DataFrame,
    crypto_prices: pd.DataFrame,
    price_col: str = "adjClose",
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Build equity, crypto, and combined return panels.

    Returns are first calculated separately inside the equity and 365-day
    crypto panels. For the combined panel, the already-calculated crypto daily
    returns are left-aligned to the equity-return calendar. This follows the
    Project Brief and intentionally excludes weekend-only crypto moves.
    """
    eq = _clean_price_panel(equity_prices, price_col)
    cr = _clean_price_panel(crypto_prices, price_col)

    eq_ret = _wide_returns(eq, price_col)
    cr_ret = _wide_returns(cr, price_col)

    # Prefix crypto columns only if a crypto ticker overlaps with an equity ticker.
    overlap = set(eq_ret.columns).intersection(set(cr_ret.columns))
    if overlap:
        cr_ret = cr_ret.rename(columns={c: f"CRYPTO_{c}" for c in cr_ret.columns})

    # Brief rule: compute returns inside each native panel first, then left-merge
    # crypto returns onto the equity trading calendar. Do not compound weekend
    # moves into Monday and do not merge price levels before differencing.
    crypto_on_equity_calendar = cr_ret.reindex(eq_ret.index)
    combined = pd.concat([eq_ret, crypto_on_equity_calendar], axis=1)

    panels = {
        "equity": eq_ret.replace([np.inf, -np.inf], np.nan),
        "crypto": cr_ret.replace([np.inf, -np.inf], np.nan),
        "combined": combined.replace([np.inf, -np.inf], np.nan),
    }

    eq_meta = (
        eq[["ticker", "sector"]]
        .drop_duplicates()
        .rename(columns={"ticker": "asset"})
        .assign(asset_class="Equity")
    )

    cr_assets = list(cr_ret.columns)

    cr_meta = pd.DataFrame(
        {
            "asset": cr_assets,
            "sector": "Crypto",
            "asset_class": "Crypto",
        }
    )

    meta = pd.concat([eq_meta, cr_meta], ignore_index=True)
    meta = meta.drop_duplicates("asset").sort_values(["asset_class", "sector", "asset"])

    return panels, meta


def price_panel_to_returns(
    equity_prices: pd.DataFrame,
    crypto_prices: pd.DataFrame,
    price_col: str = "adjClose",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backward-compatible wrapper returning the combined equity-calendar panel."""
    panels, meta = build_return_panels(equity_prices, crypto_prices, price_col)
    return panels["combined"], meta


def _shrink_cov(cov: pd.DataFrame, shrink: float = 0.10) -> np.ndarray:
    """Simple diagonal shrinkage to stabilise short-window covariance."""
    c = cov.fillna(0.0).to_numpy(dtype=float)
    n = c.shape[0]
    avg_var = np.trace(c) / n if n else 0.0
    target = np.eye(n) * avg_var
    out = (1.0 - shrink) * c + shrink * target
    out = out + np.eye(n) * 1e-8
    return out


def _normalise_weights(w: np.ndarray, max_weight: float) -> np.ndarray:
    """Clean numerical noise and enforce a simple long-only cap approximately."""
    w = np.asarray(w, dtype=float)
    w[~np.isfinite(w)] = 0.0
    w = np.maximum(w, 0.0)

    if w.sum() <= 0:
        w = np.ones_like(w) / len(w)
    else:
        w = w / w.sum()

    # Iterative cap-and-redistribute.
    max_weight = max(max_weight, 1.0 / len(w))
    for _ in range(20):
        over = w > max_weight
        if not over.any():
            break
        excess = (w[over] - max_weight).sum()
        w[over] = max_weight
        under = ~over
        if under.any() and w[under].sum() > 0:
            w[under] += excess * w[under] / w[under].sum()
        else:
            break

    return w / w.sum()


def _solve_long_only_minimise(
    obj,
    n: int,
    max_weight: float,
    group_indices: np.ndarray | None = None,
    group_cap: float | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Solve a long-only constrained optimisation and report solver quality."""
    max_weight = max(max_weight, 1.0 / n)
    x0 = np.ones(n) / n

    try:
        from scipy.optimize import minimize

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        if group_indices is not None and group_cap is not None:
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda w: float(group_cap - np.sum(w[group_indices])),
                }
            )

        result = minimize(
            obj,
            x0,
            method="SLSQP",
            bounds=[(0.0, max_weight)] * n,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-10, "disp": False},
        )
        if result.success and np.all(np.isfinite(result.x)):
            weights = _normalise_weights(result.x, max_weight)
            weights = _enforce_group_cap(
                weights,
                group_indices=group_indices,
                group_cap=group_cap,
                max_weight=max_weight,
            )
            return weights, {
                "solver_success": True,
                "used_fallback": False,
                "solver_status": int(result.status),
                "solver_message": str(result.message),
                "objective_value": float(obj(weights)),
                "iterations": int(getattr(result, "nit", 0)),
            }
        message = str(result.message)
        status = int(result.status)
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        status = -1

    fallback = _normalise_weights(x0, max_weight)
    fallback = _enforce_group_cap(
        fallback,
        group_indices=group_indices,
        group_cap=group_cap,
        max_weight=max_weight,
    )
    return fallback, {
        "solver_success": False,
        "used_fallback": True,
        "solver_status": status,
        "solver_message": message,
        "objective_value": float(obj(fallback)),
        "iterations": 0,
    }


def _enforce_group_cap(
    weights: np.ndarray,
    group_indices: np.ndarray | None,
    group_cap: float | None,
    max_weight: float,
) -> np.ndarray:
    """Project weights onto an optional aggregate sleeve cap."""
    w = np.asarray(weights, dtype=float).copy()
    if group_indices is None or group_cap is None or len(group_indices) == 0:
        return w / w.sum()

    group_indices = np.asarray(group_indices, dtype=int)
    outside = np.setdiff1d(np.arange(len(w)), group_indices)
    if len(outside) == 0 and group_cap < 1.0 - 1e-12:
        raise ValueError("Group cap is infeasible because the group contains all assets.")

    group_total = float(w[group_indices].sum())
    if group_total <= group_cap + 1e-12:
        return w / w.sum()

    excess = group_total - float(group_cap)
    if group_total > 0:
        w[group_indices] *= float(group_cap) / group_total
    room = np.maximum(float(max_weight) - w[outside], 0.0)
    if room.sum() + 1e-12 < excess:
        raise ValueError("Group cap is infeasible under the asset-level weight cap.")
    w[outside] += excess * room / room.sum()
    return w / w.sum()


def _solve_min_cvar(
    returns: np.ndarray,
    max_weight: float,
    confidence: float,
    group_indices: np.ndarray | None = None,
    group_cap: float | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Minimise historical Expected Shortfall using the Rockafellar-Uryasev LP."""
    from scipy.optimize import linprog

    observations, n = returns.shape
    if observations < 20:
        raise ValueError("Minimum-CVaR requires at least 20 observations.")
    if not 0.50 < confidence < 1.0:
        raise ValueError("CVaR confidence must lie between 0.50 and 1.00.")

    # Variables are [asset weights, VaR loss threshold, tail slacks].
    objective = np.concatenate(
        [
            np.zeros(n),
            np.array([1.0]),
            np.full(observations, 1.0 / ((1.0 - confidence) * observations)),
        ]
    )
    a_ub = np.zeros((observations, n + 1 + observations))
    a_ub[:, :n] = -returns
    a_ub[:, n] = -1.0
    a_ub[:, n + 1 :] = -np.eye(observations)
    b_ub = np.zeros(observations)

    if group_indices is not None and group_cap is not None:
        group_row = np.zeros((1, n + 1 + observations))
        group_row[0, group_indices] = 1.0
        a_ub = np.vstack([a_ub, group_row])
        b_ub = np.concatenate([b_ub, [float(group_cap)]])

    a_eq = np.zeros((1, n + 1 + observations))
    a_eq[0, :n] = 1.0
    bounds = [(0.0, float(max_weight))] * n + [(None, None)] + [
        (0.0, None)
    ] * observations

    result = linprog(
        objective,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=np.array([1.0]),
        bounds=bounds,
        method="highs",
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"Minimum-CVaR solver failed: {result.message}")

    weights = _normalise_weights(result.x[:n], max_weight)
    weights = _enforce_group_cap(
        weights,
        group_indices=group_indices,
        group_cap=group_cap,
        max_weight=max_weight,
    )
    return weights, {
        "solver_success": True,
        "used_fallback": False,
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "objective_value": float(result.fun),
        "iterations": int(getattr(result, "nit", 0)),
    }


def optimise_weights(
    window_returns: pd.DataFrame,
    method: str,
    max_weight: float = 0.20,
    rf_annual: float = 0.00,
    periods_per_year: int = EQUITY_TRADING_DAYS,
    group_cap_assets: list[str] | None = None,
    group_cap: float | None = None,
    cvar_confidence: float = 0.95,
    return_diagnostics: bool = False,
) -> pd.Series | tuple[pd.Series, dict[str, object]]:
    """Estimate long-only portfolio weights from a trailing return window."""
    data = window_returns.dropna(axis=1, thresh=int(0.80 * len(window_returns))).fillna(0.0)
    data = data.loc[:, data.std() > 0]

    if data.empty:
        raise ValueError("No valid assets available for optimisation.")

    assets = list(data.columns)
    n = len(assets)
    group_indices = None
    if group_cap_assets:
        group_set = set(group_cap_assets)
        group_indices = np.array(
            [i for i, asset in enumerate(assets) if asset in group_set],
            dtype=int,
        )

    mu = data.mean().to_numpy(dtype=float) * periods_per_year
    cov = _shrink_cov(data.cov() * periods_per_year)

    if method == "equal_weight":
        w = np.ones(n) / n
        diagnostics = {
            "solver_success": True,
            "used_fallback": False,
            "solver_status": 0,
            "solver_message": "Deterministic equal-weight benchmark; no solver used.",
            "objective_value": np.nan,
            "iterations": 0,
        }

    elif method == "min_vol":
        def obj(w):
            return float(w @ cov @ w)

        w, diagnostics = _solve_long_only_minimise(
            obj,
            n=n,
            max_weight=max_weight,
            group_indices=group_indices,
            group_cap=group_cap,
        )

    elif method == "tangency":
        rf = rf_annual

        def obj(w):
            port_ret = float(w @ mu)
            port_var = float(w @ cov @ w)
            port_vol = math.sqrt(max(port_var, 1e-12))
            return -((port_ret - rf) / port_vol)

        w, diagnostics = _solve_long_only_minimise(
            obj,
            n=n,
            max_weight=max_weight,
            group_indices=group_indices,
            group_cap=group_cap,
        )

    elif method == "risk_parity":
        def obj(w):
            marginal = cov @ w
            contributions = w * marginal
            total = float(contributions.sum())
            if total <= 1e-16:
                return 1e6
            target = total / n
            return float(np.sum((contributions - target) ** 2) / (total**2))

        w, diagnostics = _solve_long_only_minimise(
            obj,
            n=n,
            max_weight=max_weight,
            group_indices=group_indices,
            group_cap=group_cap,
        )

    elif method == "min_cvar":
        w, diagnostics = _solve_min_cvar(
            data.to_numpy(dtype=float),
            max_weight=max_weight,
            confidence=cvar_confidence,
            group_indices=group_indices,
            group_cap=group_cap,
        )

    else:
        raise ValueError(f"Unknown optimisation method: {method}")

    weight_series = pd.Series(w, index=assets, name="weight")
    diagnostics.update(
        {
            "method": method,
            "n_assets": n,
            "max_weight_observed": float(weight_series.max()),
            "weight_sum": float(weight_series.sum()),
            "group_cap": group_cap,
            "group_weight_observed": (
                float(weight_series.loc[weight_series.index.intersection(group_cap_assets)].sum())
                if group_cap_assets
                else np.nan
            ),
            "cvar_confidence": cvar_confidence if method == "min_cvar" else np.nan,
        }
    )
    if return_diagnostics:
        return weight_series, diagnostics
    return weight_series


def _monthly_rebalance_dates(returns: pd.DataFrame, window: int) -> list[pd.Timestamp]:
    """First available trading day of each month after the initial window."""
    dates = pd.Series(returns.index[window:])
    return list(dates.groupby(dates.dt.to_period("M")).first())


def walk_forward_backtest(
    returns: pd.DataFrame,
    asset_meta: pd.DataFrame,
    fund_name: str,
    assets: list[str],
    method: str,
    window: int = 252,
    max_weight: float = 0.20,
    rf_annual: float = 0.00,
    periods_per_year: int = EQUITY_TRADING_DAYS,
    sentiment_index: pd.DataFrame | None = None,
    sentiment_tilt: bool = False,
    group_cap_assets: list[str] | None = None,
    group_cap: float | None = None,
    cvar_confidence: float = 0.95,
) -> BacktestResult:
    """Run a monthly walk-forward out-of-sample fund backtest."""
    usable_assets = [a for a in assets if a in returns.columns]
    if not usable_assets:
        raise ValueError(f"No usable assets for fund {fund_name}")

    returns = returns.sort_index()
    rebalance_dates = _monthly_rebalance_dates(returns, window=window)

    fund_returns = pd.Series(index=returns.index[window:], dtype=float, name=fund_name)
    weight_rows = []
    diagnostic_rows = []

    sentiment_lookup = None
    if sentiment_tilt and sentiment_index is not None:
        si = sentiment_index.copy()
        si["date"] = pd.to_datetime(si["date"]).dt.normalize()
        sentiment_lookup = si.pivot(
            index="date",
            columns="sector",
            values="sentiment_signal_z",
        )

    for i, reb_date in enumerate(rebalance_dates):
        # Use returns strictly before the rebalance date.
        hist = returns.loc[returns.index < reb_date, usable_assets].tail(window)

        weights, diagnostics = optimise_weights(
            hist,
            method=method,
            max_weight=max_weight,
            rf_annual=rf_annual,
            periods_per_year=periods_per_year,
            group_cap_assets=group_cap_assets,
            group_cap=group_cap,
            cvar_confidence=cvar_confidence,
            return_diagnostics=True,
        )

        if sentiment_tilt and sentiment_lookup is not None and reb_date in sentiment_lookup.index:
            sector_signal = sentiment_lookup.loc[reb_date]
            weights = apply_sector_sentiment_tilt(
                weights=weights,
                asset_meta=asset_meta,
                sector_signal=sector_signal,
                tilt_strength=0.30,
                max_weight=max_weight,
            )

        diagnostics.update(
            {
                "date": reb_date,
                "fund": fund_name,
                "window_observations": int(len(hist)),
                "sentiment_tilt": bool(sentiment_tilt),
                "post_tilt_max_weight": float(weights.max()),
                "post_tilt_weight_sum": float(weights.sum()),
            }
        )
        diagnostic_rows.append(diagnostics)

        next_date = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else returns.index[-1] + pd.Timedelta(days=1)
        period = returns.loc[(returns.index >= reb_date) & (returns.index < next_date), weights.index]

        fund_returns.loc[period.index] = period.fillna(0.0).dot(weights)

        meta = asset_meta.set_index("asset")
        for asset, weight in weights.items():
            sector = meta.loc[asset, "sector"] if asset in meta.index else ""
            asset_class = meta.loc[asset, "asset_class"] if asset in meta.index else ""
            weight_rows.append(
                {
                    "date": reb_date,
                    "fund": fund_name,
                    "asset": asset,
                    "weight": float(weight),
                    "asset_class": asset_class,
                    "sector": sector,
                    "method": method,
                    "sentiment_tilt": bool(sentiment_tilt),
                }
            )

    weights_df = pd.DataFrame(weight_rows)
    return BacktestResult(
        returns=fund_returns.dropna(),
        weights=weights_df,
        diagnostics=pd.DataFrame(diagnostic_rows),
    )


def performance_metrics(
    fund_returns: pd.DataFrame,
    rf_annual: float = 0.00,
    periods_per_year: int | dict[str, int] = EQUITY_TRADING_DAYS,
) -> pd.DataFrame:
    """Compute standard fund fact-sheet metrics."""
    rows = []

    for fund in fund_returns.columns:
        r = fund_returns[fund].dropna()
        if r.empty:
            continue

        annualisation = (
            int(periods_per_year[fund])
            if isinstance(periods_per_year, dict)
            else int(periods_per_year)
        )

        growth = (1.0 + r).cumprod()
        running_max = growth.cummax()
        drawdown = growth / running_max - 1.0

        ann_return = growth.iloc[-1] ** (annualisation / len(r)) - 1.0
        ann_vol = r.std() * math.sqrt(annualisation)
        rf_daily = (1.0 + rf_annual) ** (1.0 / annualisation) - 1.0
        sharpe = (
            (r.mean() - rf_daily) / r.std() * math.sqrt(annualisation)
            if r.std() > 0
            else np.nan
        )
        downside = np.minimum(r - rf_daily, 0.0)
        downside_deviation = math.sqrt(float(np.mean(np.square(downside))))
        sortino = (
            (r.mean() - rf_daily) / downside_deviation * math.sqrt(annualisation)
            if downside_deviation > 0
            else np.nan
        )
        lower_quantile = float(r.quantile(0.05))
        tail = r.loc[r <= lower_quantile]
        var_95 = -lower_quantile
        cvar_95 = -float(tail.mean()) if not tail.empty else np.nan

        rows.append(
            {
                "fund": fund,
                "start_date": r.index.min().date(),
                "end_date": r.index.max().date(),
                "observations": int(len(r)),
                "periods_per_year": annualisation,
                "annual_return": float(ann_return),
                "annual_volatility": float(ann_vol),
                "sharpe_ratio": float(sharpe),
                "sortino_ratio": float(sortino),
                "var_95": float(var_95),
                "cvar_95": float(cvar_95),
                "max_drawdown": float(drawdown.min()),
                "total_return": float(growth.iloc[-1] - 1.0),
            }
        )

    return pd.DataFrame(rows).sort_values("sharpe_ratio", ascending=False).reset_index(drop=True)
