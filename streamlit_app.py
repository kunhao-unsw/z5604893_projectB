"""MarketPulse Funds - Streamlit app.

The app reads precomputed Project B outputs from results/.
It does not rerun VADER, backtests, or raw-data processing.
"""
from __future__ import annotations

import pathlib

import pandas as pd
import streamlit as st

from src.analytics import (
    align_fund_returns_for_allocation,
    apply_management_fee,
    portfolio_summary,
    worst_rolling_return,
)


ROOT = pathlib.Path(__file__).resolve().parent
RESULTS_DATA = ROOT / "results" / "data"
RESULTS_TABLES = ROOT / "results" / "tables"


st.set_page_config(page_title="MarketPulse Funds", layout="wide")
st.markdown(
    """
    <style>
    /*
    Use Streamlit theme variables rather than fixed light colours. This keeps
    text and backgrounds readable when the viewer switches to Dark mode.
    */
    .stApp {
        background: var(--background-color);
        color: var(--text-color);
    }
    [data-testid="stMetric"] {
        background: var(--secondary-background-color);
        color: var(--text-color);
        border: 1px solid rgba(127, 127, 127, 0.25);
        border-radius: 12px;
        padding: 14px;
    }
    h1, h2, h3 {color: var(--text-color); letter-spacing: -0.02em;}
    .mp-kicker {
        color: #14b8a6;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }
    .mp-panel {
        background: var(--secondary-background-color);
        border-left: 4px solid #14b8a6;
        border-radius: 10px;
        padding: 0.9rem 1rem;
        margin: 0.4rem 0 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("MarketPulse Funds")
st.caption(
    "Systematic equity, crypto and combined funds with finance-aware news "
    "sentiment. Research prototype—not financial advice."
)
st.markdown(
    '<div class="mp-kicker">Evidence first · downside aware · fully precomputed</div>',
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_fund_returns() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DATA / "fund_returns.csv", parse_dates=["date"])
    return df.set_index("date").sort_index()


@st.cache_data(show_spinner=False)
def load_fund_weights() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DATA / "fund_weights.csv", parse_dates=["date"])
    return df.sort_values(["date", "fund", "weight"], ascending=[True, True, False])


@st.cache_data(show_spinner=False)
def load_metrics() -> pd.DataFrame:
    return pd.read_csv(RESULTS_TABLES / "performance_metrics.csv")


@st.cache_data(show_spinner=False)
def load_sentiment() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DATA / "sector_sentiment_index.csv", parse_dates=["date"])
    return df.sort_values(["date", "sector"])


@st.cache_data(show_spinner=False)
def load_fusion_comparison() -> pd.DataFrame:
    return pd.read_csv(RESULTS_TABLES / "fusion_before_after.csv")


@st.cache_data(show_spinner=False)
def load_fusion_robustness() -> pd.DataFrame:
    return pd.read_csv(RESULTS_TABLES / "fusion_robustness.csv")


def growth_of_one(returns: pd.DataFrame) -> pd.DataFrame:
    return (1.0 + returns).cumprod()


def drawdown_table(returns: pd.DataFrame) -> pd.DataFrame:
    growth = growth_of_one(returns)
    return growth / growth.cummax() - 1.0


try:
    fund_returns = load_fund_returns()
    fund_weights = load_fund_weights()
    metrics = load_metrics()
    sentiment = load_sentiment()
    fusion_comparison = load_fusion_comparison()
    fusion_robustness = load_fusion_robustness()
except FileNotFoundError as exc:
    st.error(
        "Required results files are missing. Run `python scripts/run_part_b.py` "
        "before launching the app."
    )
    st.exception(exc)
    st.stop()


tab_overview, tab_fact_sheet, tab_allocation, tab_sentiment, tab_innovation, tab_method = st.tabs(
    [
        "Compare Funds",
        "Fund Fact Sheets",
        "Build an Allocation",
        "News Pulse",
        "Downside Innovation",
        "Method Notes",
    ]
)


with tab_overview:
    st.subheader("Out-of-sample fund comparison")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Funds", len(fund_returns.columns))
    c2.metric("Backtest start", str(fund_returns.index.min().date()))
    c3.metric("Backtest end", str(fund_returns.index.max().date()))
    c4.metric("Calendar observations", f"{len(fund_returns):,}")

    st.markdown(
        '<div class="mp-panel">Use the same evidence sequence as a fund '
        'fact sheet: return, volatility, downside risk, drawdown and holdings. '
        'Crypto-only funds use 365-day annualisation; equity and combined funds '
        'use 252.</div>',
        unsafe_allow_html=True,
    )
    family = st.segmented_control(
        "Fund universe",
        ["All", "Equity", "Crypto", "Combined"],
        default="All",
    )
    display_metrics = metrics.copy()
    display_metrics["Universe"] = display_metrics["fund"].map(
        lambda name: (
            "Crypto" if name.startswith("Crypto ")
            else "Combined" if name.startswith("Combined ") or name == "MarketPulse TailGuard"
            else "Equity"
        )
    )
    if family != "All":
        display_metrics = display_metrics.loc[display_metrics["Universe"].eq(family)]

    percent_columns = [
        "annual_return",
        "annual_volatility",
        "var_95",
        "cvar_95",
        "max_drawdown",
        "total_return",
    ]
    display_metrics[percent_columns] = display_metrics[percent_columns] * 100.0
    display_metrics = display_metrics.rename(
        columns={
            "fund": "Fund",
            "annual_return": "Annual return (%)",
            "annual_volatility": "Volatility (%)",
            "sharpe_ratio": "Sharpe",
            "sortino_ratio": "Sortino",
            "var_95": "95% VaR loss (%)",
            "cvar_95": "95% CVaR loss (%)",
            "max_drawdown": "Max drawdown (%)",
            "total_return": "Total return (%)",
        }
    )
    display_columns = [
        "Fund", "Universe", "Annual return (%)", "Volatility (%)", "Sharpe",
        "Sortino", "95% CVaR loss (%)", "Max drawdown (%)", "Total return (%)",
    ]
    st.dataframe(
        display_metrics[display_columns].style.format(
            {
                "Annual return (%)": "{:.2f}",
                "Volatility (%)": "{:.2f}",
                "Sharpe": "{:.2f}",
                "Sortino": "{:.2f}",
                "95% CVaR loss (%)": "{:.2f}",
                "Max drawdown (%)": "{:.2f}",
                "Total return (%)": "{:.2f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    selected_names = display_metrics["Fund"].tolist()
    st.markdown("### Growth of $1")
    st.line_chart(growth_of_one(fund_returns[selected_names]))
    st.markdown("### Drawdowns")
    st.line_chart(drawdown_table(fund_returns[selected_names]))


with tab_fact_sheet:
    st.subheader("Fund fact sheet")
    fund = st.selectbox("Choose a fund", list(fund_returns.columns))
    row = metrics.loc[metrics["fund"] == fund].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Annual return", f"{row['annual_return']:.2%}")
    c2.metric("Annual volatility", f"{row['annual_volatility']:.2%}")
    c3.metric("Sharpe ratio", f"{row['sharpe_ratio']:.2f}")
    c4.metric("Sortino ratio", f"{row['sortino_ratio']:.2f}")
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("95% CVaR loss", f"{row['cvar_95']:.2%}")
    c6.metric("Max drawdown", f"{row['max_drawdown']:.2%}")
    c7.metric("Total return", f"{row['total_return']:.2%}")
    c8.metric("Observations", f"{int(row['observations']):,}")

    st.markdown("### Growth and drawdown")
    left, right = st.columns(2)
    left.line_chart(growth_of_one(fund_returns[[fund]].dropna()))
    right.line_chart(drawdown_table(fund_returns[[fund]].dropna()))

    selected_weights = fund_weights[fund_weights["fund"] == fund]
    latest_date = selected_weights["date"].max()
    latest_holdings = selected_weights[selected_weights["date"].eq(latest_date)].copy()
    latest_holdings = latest_holdings.sort_values("weight", ascending=False)
    latest_holdings["weight"] = latest_holdings["weight"] * 100.0
    st.markdown("### Latest holdings")
    st.caption(f"Target weights from the {latest_date.date()} rebalance.")
    st.dataframe(
        latest_holdings[
            ["asset", "asset_class", "sector", "weight", "method", "sentiment_tilt"]
        ].rename(
            columns={
                "asset": "Asset",
                "asset_class": "Asset class",
                "sector": "Sector",
                "weight": "Weight (%)",
                "method": "Method",
                "sentiment_tilt": "Sentiment tilt",
            }
        ).style.format({"Weight (%)": "{:.2f}"}),
        width="stretch",
        hide_index=True,
    )

    if selected_weights["asset_class"].nunique() > 1:
        st.markdown("### Asset-class weight history")
        weight_history = (
            selected_weights.groupby(["date", "asset_class"], as_index=False)["weight"]
            .sum()
            .pivot(index="date", columns="asset_class", values="weight")
            .fillna(0.0)
        )
    else:
        st.markdown("### Top holding weight history")
        top_assets = selected_weights.groupby("asset")["weight"].mean().nlargest(10).index
        weight_history = (
            selected_weights[selected_weights["asset"].isin(top_assets)]
            .pivot(index="date", columns="asset", values="weight")
            .fillna(0.0)
        )
    st.line_chart(weight_history)


with tab_allocation:
    st.subheader("Hypothetical allocation and fee lab")
    st.write(
        "Select funds and weights, then compare gross and fee-adjusted historical "
        "outcomes. This is an educational research tool, not personal advice."
    )
    chosen_funds = st.multiselect(
        "Select funds",
        list(fund_returns.columns),
        default=list(fund_returns.columns[:3]),
    )

    if not chosen_funds:
        st.warning("Select at least one fund.")
    else:
        raw_weights = {}
        for fund_name in chosen_funds:
            raw_weights[fund_name] = st.slider(
                fund_name, min_value=0, max_value=100, value=10, step=5
            )
        total = sum(raw_weights.values())
        if total <= 0:
            st.warning("Allocation weights cannot all be zero.")
        else:
            allocation = pd.Series(raw_weights, dtype=float) / total
            aligned_returns, periods = align_fund_returns_for_allocation(
                fund_returns,
                chosen_funds,
            )
            portfolio_returns = aligned_returns.dot(allocation)

            input_left, input_right = st.columns(2)
            initial_investment = input_left.number_input(
                "Hypothetical starting amount ($)",
                min_value=100.0,
                max_value=1_000_000.0,
                value=10_000.0,
                step=500.0,
            )
            annual_fee_percent = input_right.slider(
                "Annual management fee (%)",
                min_value=0.0,
                max_value=2.0,
                value=0.75,
                step=0.05,
            )
            net_returns = apply_management_fee(
                portfolio_returns,
                annual_fee=annual_fee_percent / 100.0,
                periods_per_year=periods,
            )

            allocation_table = allocation.rename("weight").reset_index()
            allocation_table.columns = ["fund", "weight"]
            allocation_table["weight"] = allocation_table["weight"] * 100.0
            st.dataframe(
                allocation_table.rename(
                    columns={"fund": "Fund", "weight": "Allocation (%)"}
                ).style.format({"Allocation (%)": "{:.2f}"}),
                width="stretch",
                hide_index=True,
            )

            value_paths = pd.DataFrame(
                {
                    "Gross value": initial_investment * (1.0 + portfolio_returns).cumprod(),
                    "After management fee": initial_investment * (1.0 + net_returns).cumprod(),
                }
            )
            st.line_chart(value_paths)
            gross_summary = portfolio_summary(portfolio_returns, periods)
            gross_final = float(value_paths["Gross value"].iloc[-1])
            net_final = float(value_paths["After management fee"].iloc[-1])
            fee_drag = gross_final - net_final
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Gross ending value", f"${gross_final:,.0f}")
            c2.metric("Net ending value", f"${net_final:,.0f}")
            c3.metric("Management-fee drag", f"${fee_drag:,.0f}")
            c4.metric("Gross amount earned", f"${gross_final - initial_investment:,.0f}")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Annual return", f"{gross_summary['annual_return']:.2%}")
            r2.metric("Sortino ratio", f"{gross_summary['sortino_ratio']:.2f}")
            r3.metric("95% CVaR loss", f"{gross_summary['cvar_95']:.2%}")
            rolling_window = 30 if periods == 365 else 21
            r4.metric(
                f"Worst {rolling_window}-period return",
                f"{worst_rolling_return(portfolio_returns, rolling_window):.2%}",
            )
            st.caption(
                f"{len(portfolio_returns):,} shared observations. All-crypto "
                "allocations use the seven-day calendar; any selection containing an "
                "equity or combined fund uses common equity-date observations. "
                "The fee illustration excludes tax, brokerage, slippage and market impact."
            )


with tab_sentiment:
    st.subheader("Finance-aware sector news pulse")
    sectors = sorted(sentiment["sector"].unique())
    selected_sectors = st.multiselect(
        "Select sectors",
        sectors,
        default=sectors[:3],
    )

    if selected_sectors:
        selected_sentiment = sentiment[sentiment["sector"].isin(selected_sectors)]
        sentiment_view = st.radio(
            "Sentiment view",
            ["0–100 level", "Standardised z-score"],
            horizontal=True,
        )
        value_column = (
            "sentiment_score_21d_avg"
            if sentiment_view == "0–100 level"
            else "sentiment_z_full_sample"
        )
        plot_df = selected_sentiment.pivot(
            index="date",
            columns="sector",
            values=value_column,
        ).sort_index()
        if sentiment_view == "0–100 level":
            st.markdown("### Sentiment score (0–100)")
            st.caption("50 is neutral; curves show a 21-trading-day average.")
        else:
            st.markdown("### Sentiment relative to each sector's history")
            st.caption(
                "Descriptive full-sample z-scores: 0 is the sector's historical "
                "mean. Trading fusion instead uses the lagged expanding z-score "
                "available at each date."
            )
        st.line_chart(plot_df)

        summary = (
            selected_sentiment.groupby("sector", as_index=False)
            .agg(
                mean_score=("sentiment_score_0_100", "mean"),
                sentiment_volatility=("sentiment_compound", "std"),
                mean_coverage_rate=("coverage_rate", "mean"),
                total_headlines=("total_headlines", "sum"),
                finance_lexicon_effect=("finance_lexicon_effect", "mean"),
            )
            .sort_values("mean_score", ascending=False)
        )
        st.markdown("### Selected-sector summary")
        summary["mean_coverage_rate"] = summary["mean_coverage_rate"] * 100.0
        summary = summary.rename(
            columns={
                "sector": "Sector",
                "mean_score": "Mean score (0–100)",
                "sentiment_volatility": "Sentiment volatility",
                "mean_coverage_rate": "News coverage (%)",
                "total_headlines": "Headlines",
                "finance_lexicon_effect": "Mean finance-lexicon effect",
            }
        )
        st.dataframe(
            summary.style.format(
                {
                    "Mean score (0–100)": "{:.2f}",
                    "Sentiment volatility": "{:.3f}",
                    "News coverage (%)": "{:.2f}",
                    "Headlines": "{:,.0f}",
                    "Mean finance-lexicon effect": "{:.4f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

        st.markdown("### Fusion: base versus sentiment tilt")
        st.dataframe(fusion_comparison.round(4), width="stretch", hide_index=True)
        fusion_returns = fund_returns[
            ["Equity Tangency", "Sentiment Tilt Equity"]
        ].dropna()
        st.line_chart(growth_of_one(fusion_returns))
        st.caption(
            "The tilted fund is shown even when it underperforms; a negative "
            "result is evidence about the limits of simple headline timing."
        )

        with st.expander("View recent sector-day data"):
            st.dataframe(selected_sentiment.tail(200), width="stretch", hide_index=True)
    else:
        st.warning("Select at least one sector.")


with tab_innovation:
    st.subheader("Downside-risk innovation and robustness")
    st.markdown(
        '<div class="mp-panel"><strong>MarketPulse TailGuard</strong> is a '
        'combined equity–crypto fund that minimises historical 95% Conditional '
        'Value at Risk (Expected Shortfall), subject to long-only weights, a 20% '
        'asset cap and a 30% aggregate crypto-sleeve cap. It is evaluated with '
        'the same monthly walk-forward design as the baseline funds.</div>',
        unsafe_allow_html=True,
    )

    comparison_names = [
        "Combined Equal Weight",
        "Combined Min Vol",
        "Combined Risk Parity",
        "MarketPulse TailGuard",
    ]
    downside_table = metrics.loc[metrics["fund"].isin(comparison_names)].copy()
    downside_table = downside_table[
        [
            "fund", "annual_return", "annual_volatility", "sortino_ratio",
            "cvar_95", "max_drawdown", "total_return",
        ]
    ]
    for column in [
        "annual_return", "annual_volatility", "cvar_95", "max_drawdown", "total_return"
    ]:
        downside_table[column] = downside_table[column] * 100.0
    downside_table = downside_table.rename(
        columns={
            "fund": "Fund",
            "annual_return": "Annual return (%)",
            "annual_volatility": "Volatility (%)",
            "sortino_ratio": "Sortino",
            "cvar_95": "95% CVaR loss (%)",
            "max_drawdown": "Max drawdown (%)",
            "total_return": "Total return (%)",
        }
    )
    st.dataframe(
        downside_table.style.format(
            {
                "Annual return (%)": "{:.2f}",
                "Volatility (%)": "{:.2f}",
                "Sortino": "{:.2f}",
                "95% CVaR loss (%)": "{:.2f}",
                "Max drawdown (%)": "{:.2f}",
                "Total return (%)": "{:.2f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "This comparison reports the realised result whether TailGuard wins or "
        "loses; the design is assessed by transparent downside-risk evidence, "
        "not by selecting a parameter after seeing the full test period."
    )

    st.markdown("### Sentiment fusion robustness")
    robustness_view = fusion_robustness[
        [
            "method",
            "difference_annual_return",
            "difference_sharpe_ratio",
            "difference_sortino_ratio",
            "difference_cvar_95",
            "difference_max_drawdown",
        ]
    ].copy()
    for column in [
        "difference_annual_return", "difference_cvar_95", "difference_max_drawdown"
    ]:
        robustness_view[column] = robustness_view[column] * 100.0
    robustness_view = robustness_view.rename(
        columns={
            "method": "Base method",
            "difference_annual_return": "Tilt − base annual return (pp)",
            "difference_sharpe_ratio": "Tilt − base Sharpe",
            "difference_sortino_ratio": "Tilt − base Sortino",
            "difference_cvar_95": "Tilt − base CVaR loss (pp)",
            "difference_max_drawdown": "Tilt − base max drawdown (pp)",
        }
    )
    st.dataframe(
        robustness_view.style.format(
            {
                "Tilt − base annual return (pp)": "{:+.2f}",
                "Tilt − base Sharpe": "{:+.2f}",
                "Tilt − base Sortino": "{:+.2f}",
                "Tilt − base CVaR loss (pp)": "{:+.2f}",
                "Tilt − base max drawdown (pp)": "{:+.2f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "The same 0.30 tilt strength is applied to all four base methods. No "
        "method-specific tuning is used, which reduces the risk of cherry-picking."
    )


with tab_method:
    st.subheader("Method and limitations")
    st.markdown(
        """
        **Funds.** Equity-only, crypto-only and combined funds are rebalanced
        monthly. Each rebalance uses only returns strictly before that date.
        Equity and combined funds use a 252-observation window; crypto uses 365.

        **Optimisation.** Equal weight is the benchmark. Minimum-volatility,
        tangency and risk-parity funds use covariance shrinkage, long-only
        weights and a 20% asset cap. TailGuard minimises historical 95% CVaR and
        adds a 30% aggregate crypto cap. Solver outcomes are saved as diagnostics;
        no silent fallback is accepted in the final artifacts.

        **Sentiment.** Each headline is scored before calculating ticker-day
        means and equal-weight sector means. Plain VADER is compared with a
        transparent finance lexicon. No-news ticker-days are neutral, and the
        app reports coverage so zero is not confused with abundant neutral news.

        **Fusion.** The trading signal is lagged one equity trading day and
        standardised causally using expanding past information. The 20% cap is
        re-applied after the tilt. The same fixed rule is checked across equal
        weight, minimum volatility, tangency and risk parity.

        **Calendar handling.** Crypto returns are first calculated on the native
        seven-day calendar. For combined funds, the already-calculated crypto
        daily returns are left-merged onto the equity trading calendar. This
        intentionally excludes weekend-only crypto moves, as specified in the
        Project Brief. The allocation lab uses the seven-day calendar only when
        every selected fund is crypto-only; mixed selections use shared equity dates.

        **Limitations.** Headline sentiment is a noisy proxy. The backtest
        excludes fees, taxes, slippage and market impact. Historical performance
        does not predict future performance.

        **Deployment.** The app reads precomputed CSV artifacts. It does not
        download data, run VADER or recompute portfolios during interaction.
        """
    )
    st.markdown("### Core model equations")
    st.latex(r"w_i^{EW}=\frac{1}{N}")
    st.latex(r"\min_w\;w^\top\Sigma_t w")
    st.latex(
        r"\max_w\;\frac{w^\top\mu_t-r_f}{\sqrt{w^\top\Sigma_t w}}"
    )
    st.latex(
        r"\min_w\;\left[\alpha+\frac{1}{(1-q)T}"
        r"\sum_{\tau=1}^{T}\max(-w^\top r_\tau-\alpha,0)\right],\;q=0.95"
    )
    st.caption(
        "All optimised funds satisfy 1′w = 1 and 0 ≤ wᵢ ≤ 0.20. The equations "
        "describe this project’s own fund rules; the research-paper equations "
        "shown in class were used only as a formatting example."
    )
