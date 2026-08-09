"""Reproduce Project B results.

Run from the project root:

    python scripts/run_part_b.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from src.etl import (
    integrity_audit,
    load_clean_crypto,
    load_clean_equities,
    load_clean_news,
)
from src.features import assemble_headline_panel
from src.sentiment import (
    finance_lexicon_audit_table,
    score_headlines,
    sector_sentiment_index,
)
from src.portfolios import (
    build_return_panels,
    walk_forward_backtest,
    performance_metrics,
    EQUITY_TRADING_DAYS,
    CRYPTO_TRADING_DAYS,
)


ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS_DATA = ROOT / "results" / "data"
RESULTS_TABLES = ROOT / "results" / "tables"
RESULTS_FIGURES = ROOT / "results" / "figures"

ASSET_CLASS_COLOURS = {
    "Equity": "#2563EB",
    "Crypto": "#F97316",
    "Combined": "#14B8A6",
}
METHOD_MARKERS = {
    "Equal Weight": "o",
    "Min Vol": "s",
    "Tangency": "^",
    "Risk Parity": "D",
    "TailGuard": "P",
    "Sentiment Tilt": "X",
}


def _fund_family(name: str) -> str:
    if name.startswith("Crypto "):
        return "Crypto"
    if name.startswith("Combined ") or name == "MarketPulse TailGuard":
        return "Combined"
    return "Equity"


def _fund_method(name: str) -> str:
    if name == "MarketPulse TailGuard":
        return "TailGuard"
    if name.startswith("Sentiment Tilt"):
        return "Sentiment Tilt"
    for method in ["Equal Weight", "Min Vol", "Tangency", "Risk Parity"]:
        if method in name:
            return method
    return name


def _ensure_dirs() -> None:
    RESULTS_DATA.mkdir(parents=True, exist_ok=True)
    RESULTS_TABLES.mkdir(parents=True, exist_ok=True)
    RESULTS_FIGURES.mkdir(parents=True, exist_ok=True)


def _save_figure(fig: plt.Figure, filename: str, dpi: int = 200) -> None:
    """Write a PNG completely before atomically replacing the final artifact."""
    destination = RESULTS_FIGURES / filename
    temporary = destination.with_name(f"{destination.stem}.writing.png")
    fig.savefig(temporary, dpi=dpi)
    temporary.replace(destination)


def _save_sentiment_figure(sector_index: pd.DataFrame) -> None:
    plot_df = sector_index.dropna(subset=["sentiment_score_21d_avg"]).copy()
    sectors = sorted(plot_df["sector"].unique())
    global_min = float(plot_df["sentiment_score_21d_avg"].min())
    global_max = float(plot_df["sentiment_score_21d_avg"].max())
    padding = max(0.5, 0.06 * (global_max - global_min))
    rows = int(np.ceil(len(sectors) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(13, 2.25 * rows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    palette = plt.get_cmap("tab10")
    for idx, (ax, sector) in enumerate(zip(axes, sectors)):
        group = plot_df.loc[plot_df["sector"].eq(sector)]
        ax.plot(
            group["date"],
            group["sentiment_score_21d_avg"],
            color=palette(idx % 10),
            linewidth=1.25,
        )
        ax.axhline(50, linewidth=0.7, color="#64748B", linestyle="--")
        ax.set_title(sector, loc="left", fontsize=10, fontweight="bold")
        ax.set_ylim(global_min - padding, global_max + padding)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.grid(alpha=0.18)
    for ax in axes[len(sectors):]:
        ax.set_visible(False)
    fig.suptitle("Finance-extended sector news sentiment, 2020–2023", fontsize=15)
    fig.supxlabel("Date")
    fig.supylabel("Sentiment score (0–100), 21-trading-day average")
    fig.tight_layout(rect=(0.03, 0.03, 1, 0.97))
    _save_figure(fig, "sector_sentiment_index.png")
    plt.close(fig)


def _save_sentiment_summary(scores: pd.DataFrame, sector_index: pd.DataFrame) -> None:
    label_summary = (
        scores.groupby(["sector", "sentiment_label"], as_index=False)
        .agg(ticker_days=("ticker", "count"), headlines=("headline_count", "sum"))
    )
    label_summary.to_csv(RESULTS_TABLES / "sentiment_label_summary.csv", index=False)

    sector_summary = (
        sector_index.groupby("sector", as_index=False)
        .agg(
            mean_sentiment=("sentiment_compound", "mean"),
            mean_sentiment_score=("sentiment_score_0_100", "mean"),
            sentiment_volatility=("sentiment_compound", "std"),
            mean_coverage_rate=("coverage_rate", "mean"),
            total_headlines=("total_headlines", "sum"),
            mean_finance_lexicon_effect=("finance_lexicon_effect", "mean"),
        )
        .sort_values("mean_sentiment", ascending=False)
    )
    sector_summary.to_csv(RESULTS_TABLES / "sentiment_sector_summary.csv", index=False)


def _save_growth_figure(fund_returns: pd.DataFrame) -> None:
    growth = (1.0 + fund_returns).cumprod()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharex=False)
    for ax, family in zip(axes, ["Equity", "Crypto", "Combined"]):
        columns = [c for c in growth.columns if _fund_family(c) == family]
        for col in columns:
            ax.plot(growth.index, growth[col], label=col, linewidth=1.35)
        ax.set_title(f"{family} funds")
        ax.set_xlabel("Date")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.grid(alpha=0.18)
        ax.legend(fontsize=7, frameon=False)
    axes[0].set_ylabel("Growth of $1")
    fig.suptitle("Out-of-sample growth of $1 by fund universe, 2021–2023", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save_figure(fig, "fund_growth_1dollar.png")
    plt.close(fig)


def _save_drawdown_figure(fund_returns: pd.DataFrame) -> None:
    growth = (1.0 + fund_returns).cumprod()
    drawdown = growth / growth.cummax() - 1.0

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharex=False)
    for ax, family in zip(axes, ["Equity", "Crypto", "Combined"]):
        columns = [c for c in drawdown.columns if _fund_family(c) == family]
        for col in columns:
            ax.plot(drawdown.index, drawdown[col], label=col, linewidth=1.25)
        ax.set_title(f"{family} funds")
        ax.set_xlabel("Date")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(alpha=0.18)
        ax.legend(fontsize=7, frameon=False)
    axes[0].set_ylabel("Drawdown from prior peak")
    fig.suptitle("Out-of-sample drawdowns by fund universe, 2021–2023", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save_figure(fig, "fund_drawdowns.png")
    plt.close(fig)


def _save_risk_return_figure(metrics: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    for _, row in metrics.iterrows():
        family = _fund_family(row["fund"])
        method = _fund_method(row["fund"])
        ax.scatter(
            row["annual_volatility"],
            row["annual_return"],
            color=ASSET_CLASS_COLOURS[family],
            marker=METHOD_MARKERS.get(method, "o"),
            s=72,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
    family_legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=colour,
               markeredgecolor="white", markersize=8, label=family)
        for family, colour in ASSET_CLASS_COLOURS.items()
    ]
    method_legend = [
        Line2D([0], [0], marker=marker, color="#475569", linestyle="none",
               markersize=7, label=method)
        for method, marker in METHOD_MARKERS.items()
    ]
    first_legend = ax.legend(
        handles=family_legend,
        title="Fund universe (colour)",
        frameon=False,
        loc="upper left",
    )
    ax.add_artist(first_legend)
    ax.legend(
        handles=method_legend,
        title="Method (marker)",
        frameon=False,
        loc="lower right",
        ncol=2,
    )
    ax.set_title("Out-of-sample annualised return versus volatility, 2021–2023")
    ax.set_xlabel("Annualised volatility")
    ax.set_ylabel("Annualised return (CAGR)")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(alpha=0.2)
    fig.tight_layout()
    _save_figure(fig, "fund_risk_return.png")
    plt.close(fig)


def _save_weight_figure(weights: pd.DataFrame) -> None:
    """Plot informative weights: asset classes for combined, holdings otherwise."""
    for fund, group in weights.groupby("fund"):
        if group["asset_class"].nunique() > 1:
            plot_group = (
                group.groupby(["date", "asset_class"], as_index=False)["weight"].sum()
            )
            wide = plot_group.pivot(
                index="date", columns="asset_class", values="weight"
            ).fillna(0.0)
            title = f"Asset-class weights: {fund}"
        else:
            average_weights = group.groupby("asset")["weight"].mean()
            top_assets = average_weights.nlargest(10).index
            plot_group = group[group["asset"].isin(top_assets)]
            wide = plot_group.pivot(
                index="date", columns="asset", values="weight"
            ).fillna(0.0)
            title = f"Top holding weights: {fund}"

        fig, ax = plt.subplots(figsize=(10, 5))
        wide.plot(ax=ax, linewidth=1.1)
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Weight")
        ax.legend(fontsize=8)
        fig.tight_layout()

        clean_name = fund.lower().replace(" ", "_").replace("-", "_")
        _save_figure(fig, f"weights_{clean_name}.png")
        plt.close(fig)


def _save_combined_method_weight_comparison(weights: pd.DataFrame) -> None:
    """Compare crypto allocation across combined-fund methods and TailGuard."""
    combined = weights[
        weights["fund"].str.startswith("Combined ")
        | weights["fund"].eq("MarketPulse TailGuard")
    ].copy()
    crypto_weight = (
        combined.assign(
            crypto_weight=lambda x: x["weight"].where(
                x["asset_class"].eq("Crypto"),
                0.0,
            )
        )
        .groupby(["date", "fund"], as_index=False)["crypto_weight"]
        .sum()
        .pivot(index="date", columns="fund", values="crypto_weight")
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    crypto_weight.plot(ax=ax, linewidth=1.6)
    ax.axhline(0.30, color="#DC2626", linestyle="--", linewidth=1.0,
               label="TailGuard crypto sleeve cap (30%)")
    ax.set_title("Combined funds: crypto sleeve across methods, 2021–2023")
    ax.set_xlabel("Rebalance date")
    ax.set_ylabel("Portfolio weight in crypto")
    ax.set_ylim(bottom=0.0)
    ax.legend(title="Fund method", fontsize=8)
    fig.tight_layout()
    _save_figure(fig, "weights_combined_methods_comparison.png")
    plt.close(fig)


def _save_latest_window_efficient_frontier(
    combined_returns: pd.DataFrame,
    fund_weights: pd.DataFrame,
) -> None:
    """Save a clearly labelled latest-window, in-sample efficient frontier."""
    from scipy.optimize import minimize

    latest_weights = fund_weights.loc[
        fund_weights["fund"].str.startswith("Combined ")
        | fund_weights["fund"].eq("MarketPulse TailGuard")
    ].copy()
    rebalance_date = latest_weights["date"].max()
    latest_weights = latest_weights.loc[latest_weights["date"].eq(rebalance_date)]
    window = combined_returns.loc[combined_returns.index < rebalance_date].tail(252)
    data = window.dropna(axis=1, thresh=int(0.80 * len(window))).fillna(0.0)
    data = data.loc[:, data.std() > 0]
    assets = list(data.columns)
    n = len(assets)
    mu = data.mean().to_numpy(dtype=float) * EQUITY_TRADING_DAYS
    raw_cov = data.cov().to_numpy(dtype=float) * EQUITY_TRADING_DAYS
    average_variance = float(np.trace(raw_cov) / n)
    cov = 0.90 * raw_cov + 0.10 * np.eye(n) * average_variance
    cov += np.eye(n) * 1e-8
    max_weight = max(0.20, 1.0 / n)

    def variance(w):
        return float(w @ cov @ w)

    bounds = [(0.0, max_weight)] * n
    base_constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    min_result = minimize(
        variance,
        np.ones(n) / n,
        method="SLSQP",
        bounds=bounds,
        constraints=base_constraints,
        options={"maxiter": 750, "ftol": 1e-11},
    )
    if not min_result.success:
        raise RuntimeError(f"Efficient-frontier minimum variance failed: {min_result.message}")

    order = np.argsort(mu)[::-1]
    max_return_weights = np.zeros(n)
    remaining = 1.0
    for idx in order:
        allocation = min(max_weight, remaining)
        max_return_weights[idx] = allocation
        remaining -= allocation
        if remaining <= 1e-12:
            break

    minimum_return = float(min_result.x @ mu)
    maximum_return = float(max_return_weights @ mu)
    targets = np.linspace(minimum_return, maximum_return, 24)
    points = []
    start = min_result.x
    for target in targets:
        constraints = base_constraints + [
            {"type": "eq", "fun": lambda w, target=target: float(w @ mu - target)}
        ]
        result = minimize(
            variance,
            start,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 750, "ftol": 1e-10},
        )
        if result.success:
            start = result.x
            points.append(
                {
                    "annual_return": float(result.x @ mu),
                    "annual_volatility": float(np.sqrt(variance(result.x))),
                    "estimation_end": rebalance_date.date(),
                    "window_observations": len(window),
                }
            )

    frontier = pd.DataFrame(points).sort_values("annual_volatility")
    frontier.to_csv(
        RESULTS_TABLES / "efficient_frontier_latest_window.csv",
        index=False,
    )

    positions = []
    for fund, group in latest_weights.groupby("fund"):
        w = group.set_index("asset")["weight"].reindex(assets).fillna(0.0).to_numpy()
        positions.append(
            {
                "fund": fund,
                "annual_return": float(w @ mu),
                "annual_volatility": float(np.sqrt(variance(w))),
            }
        )
    positions = pd.DataFrame(positions)
    positions.to_csv(
        RESULTS_TABLES / "latest_combined_fund_positions.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.plot(
        frontier["annual_volatility"],
        frontier["annual_return"],
        color=ASSET_CLASS_COLOURS["Combined"],
        linewidth=2.2,
        label="Long-only 20%-capped frontier",
    )
    label_offsets = {
        "Combined Min Vol": (8, -10),
        "MarketPulse TailGuard": (8, 8),
        "Combined Risk Parity": (8, 8),
        "Combined Equal Weight": (8, 8),
        "Combined Tangency": (8, 8),
    }
    for _, row in positions.iterrows():
        ax.scatter(row["annual_volatility"], row["annual_return"], s=65, zorder=3)
        ax.annotate(
            row["fund"],
            (row["annual_volatility"], row["annual_return"]),
            xytext=label_offsets.get(row["fund"], (8, 8)),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_title(
        "Latest-window combined-universe efficient frontier\n"
        f"252 observations ending before {rebalance_date.date()} (in-sample diagnostic)"
    )
    ax.set_xlabel("Annualised volatility")
    ax.set_ylabel("Annualised arithmetic expected return")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save_figure(fig, "efficient_frontier_latest_window.png")
    plt.close(fig)


def _save_fusion_robustness(
    base_returns: pd.DataFrame,
    tilted_results: dict[str, pd.Series],
) -> None:
    """Compare the same fixed sentiment rule across four base methods."""
    pairs = {
        "Equal Weight": ("Equity Equal Weight", "Sentiment Tilt Equal Weight"),
        "Min Vol": ("Equity Min Vol", "Sentiment Tilt Min Vol"),
        "Tangency": ("Equity Tangency", "Sentiment Tilt Equity"),
        "Risk Parity": ("Equity Risk Parity", "Sentiment Tilt Risk Parity"),
    }
    combined = base_returns.copy()
    for name, series in tilted_results.items():
        combined[name] = series

    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for ax, (method, (base_name, tilt_name)) in zip(axes.ravel(), pairs.items()):
        pair = combined[[base_name, tilt_name]].dropna()
        pair_metrics = performance_metrics(pair, periods_per_year=EQUITY_TRADING_DAYS)
        indexed = pair_metrics.set_index("fund")
        row = {"method": method, "base_fund": base_name, "tilted_fund": tilt_name}
        for metric in [
            "annual_return",
            "annual_volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "cvar_95",
            "max_drawdown",
            "total_return",
        ]:
            row[f"base_{metric}"] = float(indexed.loc[base_name, metric])
            row[f"tilted_{metric}"] = float(indexed.loc[tilt_name, metric])
            row[f"difference_{metric}"] = float(
                indexed.loc[tilt_name, metric] - indexed.loc[base_name, metric]
            )
        rows.append(row)

        growth = (1.0 + pair).cumprod()
        ax.plot(growth.index, growth[base_name], label="Base", linewidth=1.5)
        ax.plot(growth.index, growth[tilt_name], label="Sentiment tilt", linewidth=1.5)
        ax.set_title(method)
        ax.set_ylabel("Growth of $1")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.grid(alpha=0.18)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Fixed sentiment tilt across four equity base methods, 2021–2023")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save_figure(fig, "fusion_robustness.png")
    plt.close(fig)

    pd.DataFrame(rows).to_csv(
        RESULTS_TABLES / "fusion_robustness.csv",
        index=False,
    )
    pd.DataFrame(combined).to_csv(
        RESULTS_DATA / "fusion_robustness_returns.csv",
        index_label="date",
    )


def _save_sentiment_model_validation(scores: pd.DataFrame) -> None:
    """Save transparent plain-VADER versus finance-extension diagnostics."""
    rows = []
    for model, column in [
        ("Plain VADER", "base_compound"),
        ("Finance-extended VADER", "compound"),
    ]:
        values = scores[column]
        rows.append(
            {
                "model": model,
                "ticker_days": int(values.notna().sum()),
                "mean_compound": float(values.mean()),
                "median_compound": float(values.median()),
                "negative_share": float((values <= -0.05).mean()),
                "neutral_share": float(values.between(-0.05, 0.05, inclusive="neither").mean()),
                "positive_share": float((values >= 0.05).mean()),
            }
        )

    validation = pd.DataFrame(rows)
    validation["ticker_days_changed_by_extension"] = int(
        scores["finance_lexicon_effect"].abs().gt(1e-12).sum()
    )
    validation["share_changed_by_extension"] = float(
        scores["finance_lexicon_effect"].abs().gt(1e-12).mean()
    )
    validation.to_csv(
        RESULTS_TABLES / "sentiment_model_validation.csv",
        index=False,
    )


def _save_fusion_comparison(
    fund_returns: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    """Save the brief's required before-versus-after fusion evidence."""
    base_name = "Equity Tangency"
    tilt_name = "Sentiment Tilt Equity"
    comparison = metrics.loc[
        metrics["fund"].isin([base_name, tilt_name])
    ].copy()

    numeric = [
        "annual_return",
        "annual_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "var_95",
        "cvar_95",
        "max_drawdown",
        "total_return",
    ]
    base = comparison.set_index("fund").loc[base_name, numeric]
    tilt = comparison.set_index("fund").loc[tilt_name, numeric]
    difference = pd.DataFrame(
        [{"fund": "Tilt minus base", **(tilt - base).to_dict()}]
    )
    pd.concat([comparison, difference], ignore_index=True).to_csv(
        RESULTS_TABLES / "fusion_before_after.csv",
        index=False,
    )

    pair = fund_returns[[base_name, tilt_name]].dropna()
    growth = (1.0 + pair).cumprod()
    fig, ax = plt.subplots(figsize=(10, 6))
    growth.plot(ax=ax, linewidth=1.6)
    ax.set_title("Sentiment fusion: base versus tilted equity fund")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    fig.tight_layout()
    _save_figure(fig, "fusion_before_after.png")
    plt.close(fig)


def _run_sentiment(eq: pd.DataFrame, cr: pd.DataFrame, news: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    print("equities:", eq.shape, "crypto:", cr.shape, "news:", news.shape)

    equity_calendar = pd.to_datetime(eq["date"]).dt.normalize().drop_duplicates().sort_values()

    print("Assembling ticker-day headline panel...")
    headline_panel = assemble_headline_panel(news, equity_calendar)
    headline_panel.to_csv(RESULTS_DATA / "headline_panel_projectB.csv", index=False)

    print("Scoring each headline with plain and finance-extended VADER...")
    ticker_scores = score_headlines(headline_panel)
    ticker_scores.to_csv(RESULTS_DATA / "ticker_day_sentiment.csv", index=False)

    print("Building sector sentiment index...")
    sector_index = sector_sentiment_index(
        scores=ticker_scores,
        universe=universe,
        equity_calendar=equity_calendar,
        no_news="neutral",
    )
    sector_index.to_csv(RESULTS_DATA / "sector_sentiment_index.csv", index=False)

    _save_sentiment_summary(ticker_scores, sector_index)
    _save_sentiment_model_validation(ticker_scores)
    finance_lexicon_audit_table().to_csv(
        RESULTS_TABLES / "finance_lexicon_audit.csv",
        index=False,
    )
    _save_sentiment_figure(sector_index)

    return sector_index


def _run_funds(eq: pd.DataFrame, cr: pd.DataFrame, sector_index: pd.DataFrame) -> None:
    print("Building equity, crypto, and combined return panels...")
    panels, asset_meta = build_return_panels(eq, cr)

    panels["combined"].to_csv(RESULTS_DATA / "asset_returns_panel.csv")
    panels["equity"].to_csv(RESULTS_DATA / "equity_returns_panel.csv")
    panels["crypto"].to_csv(RESULTS_DATA / "crypto_returns_panel.csv")
    asset_meta.to_csv(RESULTS_DATA / "asset_metadata.csv", index=False)

    equity_assets = asset_meta.loc[asset_meta["asset_class"] == "Equity", "asset"].tolist()
    crypto_assets = asset_meta.loc[asset_meta["asset_class"] == "Crypto", "asset"].tolist()
    all_assets = equity_assets + crypto_assets

    fund_specs = [
        {
            "fund_name": "Equity Equal Weight",
            "panel": "equity",
            "assets": equity_assets,
            "method": "equal_weight",
            "window": 252,
            "periods_per_year": EQUITY_TRADING_DAYS,
            "sentiment_tilt": False,
        },
        {
            "fund_name": "Equity Min Vol",
            "panel": "equity",
            "assets": equity_assets,
            "method": "min_vol",
            "window": 252,
            "periods_per_year": EQUITY_TRADING_DAYS,
            "sentiment_tilt": False,
        },
        {
            "fund_name": "Equity Tangency",
            "panel": "equity",
            "assets": equity_assets,
            "method": "tangency",
            "window": 252,
            "periods_per_year": EQUITY_TRADING_DAYS,
            "sentiment_tilt": False,
        },
        {
            "fund_name": "Equity Risk Parity",
            "panel": "equity",
            "assets": equity_assets,
            "method": "risk_parity",
            "window": 252,
            "periods_per_year": EQUITY_TRADING_DAYS,
            "sentiment_tilt": False,
        },
        {
            "fund_name": "Crypto Equal Weight",
            "panel": "crypto",
            "assets": crypto_assets,
            "method": "equal_weight",
            "window": 365,
            "periods_per_year": CRYPTO_TRADING_DAYS,
            "sentiment_tilt": False,
        },
        {
            "fund_name": "Crypto Min Vol",
            "panel": "crypto",
            "assets": crypto_assets,
            "method": "min_vol",
            "window": 365,
            "periods_per_year": CRYPTO_TRADING_DAYS,
            "sentiment_tilt": False,
        },
        {
            "fund_name": "Crypto Tangency",
            "panel": "crypto",
            "assets": crypto_assets,
            "method": "tangency",
            "window": 365,
            "periods_per_year": CRYPTO_TRADING_DAYS,
            "sentiment_tilt": False,
        },
        {
            "fund_name": "Crypto Risk Parity",
            "panel": "crypto",
            "assets": crypto_assets,
            "method": "risk_parity",
            "window": 365,
            "periods_per_year": CRYPTO_TRADING_DAYS,
            "sentiment_tilt": False,
        },
        {
            "fund_name": "Combined Equal Weight",
            "panel": "combined",
            "assets": all_assets,
            "method": "equal_weight",
            "window": 252,
            "periods_per_year": EQUITY_TRADING_DAYS,
            "sentiment_tilt": False,
        },
        {
            "fund_name": "Combined Min Vol",
            "panel": "combined",
            "assets": all_assets,
            "method": "min_vol",
            "window": 252,
            "periods_per_year": EQUITY_TRADING_DAYS,
            "sentiment_tilt": False,
        },
        {
            "fund_name": "Combined Tangency",
            "panel": "combined",
            "assets": all_assets,
            "method": "tangency",
            "window": 252,
            "periods_per_year": EQUITY_TRADING_DAYS,
            "sentiment_tilt": False,
        },
        {
            "fund_name": "Combined Risk Parity",
            "panel": "combined",
            "assets": all_assets,
            "method": "risk_parity",
            "window": 252,
            "periods_per_year": EQUITY_TRADING_DAYS,
            "sentiment_tilt": False,
        },
        {
            "fund_name": "MarketPulse TailGuard",
            "panel": "combined",
            "assets": all_assets,
            "method": "min_cvar",
            "window": 252,
            "periods_per_year": EQUITY_TRADING_DAYS,
            "sentiment_tilt": False,
            "group_cap_assets": crypto_assets,
            "group_cap": 0.30,
            "cvar_confidence": 0.95,
        },
        {
            "fund_name": "Sentiment Tilt Equity",
            "panel": "equity",
            "assets": equity_assets,
            "method": "tangency",
            "window": 252,
            "periods_per_year": EQUITY_TRADING_DAYS,
            "sentiment_tilt": True,
        },
    ]

    fund_returns_list = []
    fund_weights_list = []
    diagnostic_list = []
    periods_by_fund = {}

    for spec in fund_specs:
        print(f"Backtesting {spec['fund_name']}...")
        result = walk_forward_backtest(
            returns=panels[spec["panel"]],
            asset_meta=asset_meta,
            fund_name=spec["fund_name"],
            assets=spec["assets"],
            method=spec["method"],
            window=spec["window"],
            max_weight=0.20,
            rf_annual=0.00,
            periods_per_year=spec["periods_per_year"],
            sentiment_index=sector_index,
            sentiment_tilt=spec["sentiment_tilt"],
            group_cap_assets=spec.get("group_cap_assets"),
            group_cap=spec.get("group_cap"),
            cvar_confidence=spec.get("cvar_confidence", 0.95),
        )

        fund_returns_list.append(result.returns)
        fund_weights_list.append(result.weights)
        diagnostic_list.append(result.diagnostics)
        periods_by_fund[spec["fund_name"]] = spec["periods_per_year"]

    fund_returns = pd.concat(
        fund_returns_list,
        axis=1,
        sort=False,
    ).dropna(how="all").sort_index()
    fund_weights = pd.concat(fund_weights_list, axis=0, ignore_index=True)
    diagnostics = pd.concat(diagnostic_list, axis=0, ignore_index=True)

    fund_returns.to_csv(RESULTS_DATA / "fund_returns.csv", index_label="date")
    fund_weights.to_csv(RESULTS_DATA / "fund_weights.csv", index=False)
    diagnostics.to_csv(RESULTS_TABLES / "optimizer_diagnostics.csv", index=False)

    metrics = performance_metrics(
        fund_returns,
        rf_annual=0.00,
        periods_per_year=periods_by_fund,
    )
    metrics.to_csv(RESULTS_TABLES / "performance_metrics.csv", index=False)

    # Robustness check: apply one fixed, lagged sentiment rule to four base
    # methods.  These analytical variants are kept separate from the core fund
    # menu so the app remains usable and the comparison is not cherry-picked.
    tilted_results: dict[str, pd.Series] = {
        "Sentiment Tilt Equity": fund_returns["Sentiment Tilt Equity"],
    }
    fusion_specs = [
        ("Sentiment Tilt Equal Weight", "equal_weight"),
        ("Sentiment Tilt Min Vol", "min_vol"),
        ("Sentiment Tilt Risk Parity", "risk_parity"),
    ]
    fusion_weight_rows = []
    for fund_name, method in fusion_specs:
        print(f"Backtesting robustness variant {fund_name}...")
        result = walk_forward_backtest(
            returns=panels["equity"],
            asset_meta=asset_meta,
            fund_name=fund_name,
            assets=equity_assets,
            method=method,
            window=252,
            max_weight=0.20,
            rf_annual=0.00,
            periods_per_year=EQUITY_TRADING_DAYS,
            sentiment_index=sector_index,
            sentiment_tilt=True,
        )
        tilted_results[fund_name] = result.returns
        fusion_weight_rows.append(result.weights)
        diagnostics = pd.concat([diagnostics, result.diagnostics], ignore_index=True)

    if fusion_weight_rows:
        pd.concat(fusion_weight_rows, ignore_index=True).to_csv(
            RESULTS_DATA / "fusion_robustness_weights.csv",
            index=False,
        )
    diagnostics.to_csv(RESULTS_TABLES / "optimizer_diagnostics.csv", index=False)

    _save_growth_figure(fund_returns)
    _save_drawdown_figure(fund_returns)
    _save_risk_return_figure(metrics)
    _save_weight_figure(fund_weights)
    _save_combined_method_weight_comparison(fund_weights)
    _save_fusion_comparison(fund_returns, metrics)
    _save_fusion_robustness(fund_returns, tilted_results)
    _save_latest_window_efficient_frontier(panels["combined"], fund_weights)

    print("\nPerformance metrics:")
    print(metrics)


def main() -> None:
    _ensure_dirs()

    print("Loading project data...")
    eq = load_clean_equities()
    cr = load_clean_crypto()
    news = load_clean_news()
    universe = (
        eq[["ticker", "sector"]]
        .drop_duplicates()
        .sort_values(["sector", "ticker"])
        .reset_index(drop=True)
    )
    integrity_audit(eq, cr, news).to_csv(
        RESULTS_TABLES / "data_integrity_audit.csv",
        index=False,
    )

    sector_index = _run_sentiment(eq, cr, news, universe)
    _run_funds(eq, cr, sector_index)

    print("\nSaved required outputs:")
    print(" - results/data/fund_returns.csv")
    print(" - results/data/fund_weights.csv")
    print(" - results/data/sector_sentiment_index.csv")
    print(" - results/tables/performance_metrics.csv")
    print("Done.")


if __name__ == "__main__":
    main()
