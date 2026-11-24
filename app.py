"""StockLens - the Streamlit dashboard.

Run with::

    streamlit run app.py

The dashboard is organised into five tabs, each answering one beginner
question:

==============  ==========================================================
Tab             Question it answers
==============  ==========================================================
Overview        What am I even looking at?
Performance     Who is performing better or worse?
Risk & Fit      Which looks stronger vs riskier for a beginner?
Forecast        What could happen next, and how confident can I be?
Glossary        What does all this jargon mean?
==============  ==========================================================
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from stocklens import config, glossary, performance, suitability, viz
from stocklens.data_loader import MarketData, load_market_data
from stocklens.forecaster import LIMITATIONS, forecast, forecast_explanation
from stocklens.indicators import indicator_panel

# --------------------------------------------------------------------------- #
# Page setup
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title=f"{config.APP_TITLE} - Beginner Stock Tracker",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .disclaimer-box {
        background-color: #FDF3E7;
        border-left: 5px solid #E67E22;
        padding: 14px 18px;
        border-radius: 6px;
        margin-bottom: 18px;
        font-size: 0.92rem;
        line-height: 1.55;
      }
      .disclaimer-title { font-weight: 700; color: #B9770E; }
      .card-note { font-size: 0.78rem; color: #6C757D; margin-top: -8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

#: A small, friendly shortlist students are likely to recognise.
CURATED_TICKERS: list[str] = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "TSLA", "KO", "PEP",
    "SPY", "QQQ", "JNJ", "XOM", "WMT", "DIS", "NFLX", "V",
]

#: Tickers used for the "example" preset button.
EXAMPLE_TICKERS: list[str] = ["AAPL", "MSFT", "SPY", "KO"]


# --------------------------------------------------------------------------- #
# Data loading (cached so the dashboard starts instantly on the second run)
# --------------------------------------------------------------------------- #


@st.cache_data(ttl=3600, show_spinner="Downloading price history...")
def cached_load(tickers: tuple[str, ...], benchmark: str, timeframe_days: int) -> MarketData:
    """Load market data, caching the result for an hour.

    Args:
        tickers: Symbols to load.
        benchmark: Benchmark symbol.
        timeframe_days: Trading days of history required.

    Returns:
        The loaded :class:`~stocklens.data_loader.MarketData`.
    """
    return load_market_data(
        tickers=list(tickers),
        benchmark=benchmark,
        timeframe_days=timeframe_days,
    )


def ticker_colour_map(tickers: list[str]) -> dict[str, str]:
    """Assign a stable colour to each ticker."""
    return {
        ticker: config.TICKER_COLORS[index % len(config.TICKER_COLORS)]
        for index, ticker in enumerate(tickers)
    }


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.image(
        "https://img.shields.io/badge/StockLens-Beginner%20Stock%20Tracker-2F6F9F?style=flat-square",
        width=260,
    )
    st.header("Settings")

    selected = st.multiselect(
        "Which stocks would you like to compare?",
        options=CURATED_TICKERS,
        default=list(config.DEFAULT_TICKERS),
        help="Pick two or more. You can add any other ticker in the box below.",
    )

    extra = st.text_input(
        "Add another ticker (optional)",
        value="",
        placeholder="e.g. BA, DIS, TSLA",
        help="Type any Yahoo Finance symbol and press Enter. It is added to your selection.",
    )
    if extra:
        cleaned_extra = extra.strip().upper()
        if cleaned_extra and cleaned_extra not in selected:
            selected = selected + [cleaned_extra]

    if st.button("Use the example set", help="AAPL, MSFT, SPY and KO"):
        st.session_state["_set_example"] = True

    benchmark = st.text_input(
        "Benchmark to compare against",
        value=config.DEFAULT_BENCHMARK,
        help="^GSPC is the S&P 500. Try ^IXIC (Nasdaq), ^DJI (Dow) or ^FTSE.",
    ).strip().upper()

    timeframe_label = st.selectbox(
        "Timeframe",
        options=list(config.TIMEFRAME_TRADING_DAYS.keys()),
        index=list(config.TIMEFRAME_TRADING_DAYS.keys()).index(config.DEFAULT_TIMEFRAME),
        help="How far back to look. Longer windows need more history to download.",
    )
    timeframe_days = config.TIMEFRAME_TRADING_DAYS[timeframe_label]

    st.divider()
    if st.button("🔄 Refresh data now", help="Clear the local cache and download again"):
        st.cache_data.clear()
        st.rerun()

    st.caption(
        "Prices come from Yahoo Finance's public chart service and are cached "
        "on this computer for a few hours."
    )

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #

st.title(f"📈 {config.APP_TITLE}")

# The disclaimer is rendered with Streamlit's own markdown support (rather
# than a raw HTML block) so the bold text and bullet points are formatted
# properly instead of showing the raw ** markers.
with st.container(border=True):
    st.markdown(f"### ⚠️ {config.DISCLAIMER_SHORT}")
    st.markdown(config.DISCLAIMER_LONG)

if not selected:
    st.info("Pick at least one ticker in the sidebar to get started.")
    st.stop()

# --------------------------------------------------------------------------- #
# Load data
# --------------------------------------------------------------------------- #

with st.spinner("Loading price history..."):
    try:
        data = cached_load(tuple(sorted(selected)), benchmark, timeframe_days)
    except Exception as exc:  # noqa: BLE001 - show a friendly message, not a stack trace
        st.error(
            f"Could not load market data: {exc}\n\n"
            "Check your internet connection, or place CSV files in the `data/user` "
            "folder (one file per ticker with a `Date` column and a `Close` column)."
        )
        st.stop()

closes_full = data.closes
window_closes = performance.select_window(closes_full, timeframe_days, data.benchmark)

if window_closes.empty:
    st.error("No usable price history was returned for this selection.")
    st.stop()

# --------------------------------------------------------------------------- #
# Warnings banner
# --------------------------------------------------------------------------- #

if data.errors:
    missing = ", ".join(f"**{symbol}** ({reason})" for symbol, reason in data.errors.items())
    st.warning(f"Some tickers could not be loaded and have been left out: {missing}")

if data.synthetic_symbols:
    st.error(
        "🧪 **Some or all of the prices below are SYNTHETIC sample data, not real "
        "market prices.** They were generated locally because the live download "
        f"failed for: {', '.join(data.synthetic_symbols)}. The analysis works exactly "
        "the same way, but do not quote the numbers as real market history."
    )

if data.warnings:
    with st.expander(f"⚠️ {len(data.warnings)} data quality note(s)", expanded=False):
        for note in data.warnings:
            st.markdown(f"- {note}")

sources = ", ".join(f"{symbol}: {source}" for symbol, source in sorted(data.sources.items()))
with st.expander("🔍 Where did this data come from?", expanded=False):
    st.markdown(sources)
    st.caption(
        f"History window: {data.start} to {data.end}. "
        f"Benchmark used: {data.benchmark}."
    )

# --------------------------------------------------------------------------- #
# Core computations
# --------------------------------------------------------------------------- #

table = performance.build_performance_table(window_closes, data.benchmark)
rebased_frame = performance.rebased_frame(window_closes)
relative_frame = performance.relative_performance_frame(window_closes, data.benchmark)
leaderboard = table[~table["Is benchmark"]] if not table.empty else pd.DataFrame()
best_ticker, worst_ticker = performance.performance_leaders(table)

price_map = {
    column: window_closes[column].dropna()
    for column in window_closes.columns
    if column != data.benchmark and not window_closes[column].dropna().empty
}
# The longer history is used only to warm up the moving averages and RSI, so
# that they are already calculated on the first day of a short window.
warmup_map = {
    column: closes_full[column].dropna()
    for column in window_closes.columns
    if column != data.benchmark and column in closes_full.columns
}
profiles = suitability.assess_many(price_map, warmup_map)

# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #

tab_overview, tab_performance, tab_risk, tab_forecast, tab_glossary = st.tabs(
    ["🏠 Overview", "📈 Performance", "🛡️ Risk & Suitability", "🔮 Forecast", "📖 Glossary & Methodology"]
)

# ------------------------------- OVERVIEW ---------------------------------- #
with tab_overview:
    st.subheader("Your selection at a glance")

    if not leaderboard.empty:
        top = leaderboard.iloc[0]
        bottom = leaderboard.iloc[-1]
        average_return = float(leaderboard["Total return (%)"].mean())
        average_vol = float(leaderboard["Volatility (%/yr)"].mean())

        card1, card2, card3, card4 = st.columns(4)
        card1.metric(
            f"Best performer",
            f"{top['Ticker']}",
            f"{top['Total return (%)']:+.1f}% over {timeframe_label}",
            help=glossary.explain("total_return"),
        )
        card2.metric(
            "Weakest performer",
            f"{bottom['Ticker']}",
            f"{bottom['Total return (%)']:+.1f}% over {timeframe_label}",
            help=glossary.explain("total_return"),
        )
        card3.metric(
            "Average return",
            f"{average_return:+.1f}%",
            f"across {len(leaderboard)} selected tickers",
            help="The simple average of each ticker's total return over the window.",
        )
        card4.metric(
            "Average volatility",
            f"{average_vol:.1f}%",
            "per year, typical daily movement scaled up",
            help=glossary.explain("volatility"),
        )

    st.markdown("#### In plain English")
    for sentence in performance.summary_sentences(table, data.benchmark):
        st.markdown(f"- {sentence}")

    st.markdown(
        f"<div class='card-note'>What does 'return' mean? "
        f"{glossary.explain('total_return')}</div>",
        unsafe_allow_html=True,
    )

    st.subheader("Performance chart")
    st.plotly_chart(
        viz.rebased_performance_figure(rebased_frame, data.benchmark),
        width="stretch",
    )
    st.caption(
        "Every line starts at 100 so they can be compared directly. "
        "The dashed grey line is the benchmark. "
        + glossary.explain("rebased")
    )

    st.subheader("Every number used on this page")
    if not table.empty:
        display = table.copy()
        for column in display.columns:
            if pd.api.types.is_float_dtype(display[column]):
                display[column] = display[column].round(2)
        st.dataframe(display, width="stretch", hide_index=True)
        st.download_button(
            "⬇️ Download this table as CSV",
            display.to_csv(index=False).encode("utf-8"),
            file_name=f"stocklens_performance_{timeframe_days}d.csv",
            mime="text/csv",
        )

# ------------------------------ PERFORMANCE -------------------------------- #
with tab_performance:
    st.subheader(f"Who is performing better or worse? ({timeframe_label})")

    if not leaderboard.empty:
        ranking = leaderboard.sort_values("Rank").copy()
        # The benchmark is excluded from the leaderboard, so renumber from 1
        # rather than leaving a gap where the benchmark used to be.
        ranking["Rank"] = range(1, len(ranking) + 1)
        st.dataframe(
            ranking[["Rank", "Ticker", "Total return (%)", "Annualised return (%/yr)",
                     "Return vs benchmark (pp)", "Volatility (%/yr)", "Max drawdown (%)"]].round(2),
            width="stretch",
            hide_index=True,
        )

        st.markdown("#### Returns at a glance")
        st.plotly_chart(viz.returns_bar_figure(table, data.benchmark), width="stretch")

        st.markdown("#### Relative to the benchmark")
        st.plotly_chart(viz.relative_performance_figure(relative_frame, data.benchmark), width="stretch")
        st.caption(
            "100 means the stock has performed exactly in line with the "
            f"{data.benchmark} since the window began. Above 100 means it is ahead "
            "of the benchmark; below 100 means it is behind. "
            + glossary.explain("relative_return")
        )

    st.markdown("#### What each column means")
    for metric_id in ("total_return", "annualised_return", "relative_return", "volatility", "max_drawdown"):
        entry = glossary.get_entry(metric_id)
        if entry:
            st.markdown(f"- **{entry['term']}** — {entry['plain']}")

# --------------------------- RISK & SUITABILITY ----------------------------- #
with tab_risk:
    st.subheader("Which looks stronger, and which looks riskier for a beginner?")

    st.info(
        "This assessment is **rule-based**, not a prediction. Every label below is "
        "produced by fixed thresholds that are printed next to the reason, so you "
        "can check the reasoning - or disagree with it. "
        "It deliberately never says 'buy' or 'sell'. "
        + glossary.explain("rule_based")
    )

    if profiles:
        profile_tickers = list(profiles.keys())
        selector = st.selectbox(
            "Choose a ticker to inspect",
            options=profile_tickers,
            index=0,
        )
        profile = profiles[selector]

        badge_left, badge_right = st.columns([1, 2])
        with badge_left:
            st.markdown(
                f"<div style='background:{profile.colour};color:#FFFFFF;padding:14px;"
                f"border-radius:8px;text-align:center;font-weight:700;font-size:1.05rem'>"
                f"{profile.emoji}<br>{profile.category}</div>",
                unsafe_allow_html=True,
            )
        with badge_right:
            st.markdown(f"**{selector}** — {profile.summary}")
            st.markdown(profile.explanation)

        st.markdown("#### The numbers behind the label")
        metrics_rows = [
            {"Measure": "Total return", "Value": f"{profile.metrics.get('total_return_pct', float('nan')):+.2f}%",
             "Plain English": glossary.explain("total_return")},
            {"Measure": "Annualised return", "Value": f"{profile.metrics.get('annualised_return_pct', float('nan')):+.2f}%/yr",
             "Plain English": glossary.explain("annualised_return")},
            {"Measure": "Annualised volatility", "Value": f"{profile.metrics.get('volatility_pct', float('nan')):.2f}%",
             "Plain English": glossary.explain("volatility")},
            {"Measure": "Maximum drawdown", "Value": f"{profile.metrics.get('max_drawdown_pct', float('nan')):.2f}%",
             "Plain English": glossary.explain("max_drawdown")},
            {"Measure": "Worst single day", "Value": f"{profile.metrics.get('worst_day_pct', float('nan')):.2f}%",
             "Plain English": glossary.explain("worst_day")},
            {"Measure": "RSI (14-day)", "Value": f"{profile.metrics.get('rsi', float('nan')):.1f}",
             "Plain English": glossary.explain("rsi")},
        ]
        st.dataframe(pd.DataFrame(metrics_rows), width="stretch", hide_index=True)

        st.markdown("#### Trend and momentum charts")
        # Indicators are calculated on the full history so the 50-day average
        # is already populated on the first day shown.
        full_panel = indicator_panel(closes_full[selector].dropna())
        panel = full_panel.loc[full_panel.index >= window_closes.index[0]]
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.plotly_chart(viz.indicator_figure(panel, selector), width="stretch")
        with chart_col2:
            st.plotly_chart(viz.rsi_figure(panel, selector), width="stretch")
        st.plotly_chart(viz.drawdown_figure(panel, selector), width="stretch")
        st.caption(
            "The shaded chart shows how far below its previous peak the price was. "
            "A deep dip means it would have been possible to hold a painful loss for "
            "a long stretch. " + glossary.explain("max_drawdown")
        )

    st.markdown("#### All selected tickers at once")
    overview_rows = []
    for ticker, profile in profiles.items():
        overview_rows.append(
            {
                "Ticker": ticker,
                "Profile": profile.category,
                "Total return": f"{profile.metrics.get('total_return_pct', float('nan')):+.1f}%",
                "Volatility (%/yr)": f"{profile.metrics.get('volatility_pct', float('nan')):.1f}%",
                "Max drawdown": f"{profile.metrics.get('max_drawdown_pct', float('nan')):.1f}%",
                "RSI": f"{profile.metrics.get('rsi', float('nan')):.0f}",
            }
        )
    if overview_rows:
        st.dataframe(pd.DataFrame(overview_rows), width="stretch", hide_index=True)

    with st.expander("Show the exact thresholds used by these rules"):
        st.markdown(
            f"""
| Rule | Threshold |
|---|---|
| Annualised volatility - high | {config.VOL_HIGH_PCT:.0f}% |
| Annualised volatility - low | {config.VOL_LOW_PCT:.0f}% |
| Maximum drawdown - severe | {config.DRAWDOWN_SEVERE_PCT:.0f}% |
| Maximum drawdown - caution | {config.DRAWDOWN_CAUTION_PCT:.0f}% |
| Momentum return | {config.RETURN_MOMENTUM_PCT:.0f}% |
| RSI overbought / oversold | {config.RSI_OVERBOUGHT:.0f} / {config.RSI_OVERSOLD:.0f} |
"""
        )

# --------------------------------- FORECAST --------------------------------- #
with tab_forecast:
    st.subheader("What might happen next, and how confident can we be?")

    st.warning(
        "A forecast is a **probabilistic estimate, not a guarantee**. The shaded "
        "band shows the range the price could plausibly fall inside; real prices "
        "escape it regularly. " + glossary.explain("prediction_interval")
    )

    forecast_tickers = list(price_map.keys())
    if forecast_tickers:
        forecast_selector = st.selectbox(
            "Choose a ticker to forecast",
            options=forecast_tickers,
            index=0,
            help="The model uses the last few days' returns plus a few technical indicators.",
        )

        # The model is trained on all available history, which is always more
        # data than the display window - this is stated in the UI below.
        history = closes_full[forecast_selector].dropna()

        with st.spinner("Training and testing the model..."):
            result = forecast(history, ticker=forecast_selector)

        if not result.is_valid:
            for error in result.errors:
                st.warning(error)
        else:
            metric_cols = st.columns(4)
            mae = result.test_metrics.get("MAE ($)", float("nan"))
            rmse = result.test_metrics.get("RMSE ($)", float("nan"))
            base_rmse = result.baseline_metrics.get("RMSE ($)", float("nan"))
            metric_cols[0].metric("MAE (test days)", f"${mae:,.2f}", help=glossary.explain("mae"))
            metric_cols[1].metric("RMSE (test days)", f"${rmse:,.2f}", help=glossary.explain("rmse"))
            metric_cols[2].metric(
                "Direction accuracy",
                f"{result.direction_accuracy:.0f}%",
                help=glossary.explain("direction_accuracy"),
            )
            metric_cols[3].metric(
                "Beats naive baseline?",
                "Yes" if result.beats_baseline else "No",
                f"baseline RMSE ${base_rmse:,.2f}",
                help=glossary.explain("baseline"),
            )

            st.plotly_chart(
                viz.forecast_figure(result, history, forecast_selector),
                width="stretch",
            )
            st.caption(
                f"The model was trained on the {len(history)} most recent days of "
                "available history (which is longer than the display window, because "
                "more data makes a better test of the method)."
            )

            st.markdown("#### What the model is saying")
            for sentence in forecast_explanation(result):
                st.markdown(f"- {sentence}")

            st.markdown("#### How it was tested")
            st.markdown(
                f"""
- **Split:** the oldest {result.train_size} days were used for training and the
  most recent {result.test_size} days were held back and never seen during
  training. {glossary.explain('train_test_split')}
- **Inputs:** {", ".join(result.feature_names)}. {glossary.explain('features')}
- **Model:** {result.model_name}. {glossary.explain('ridge')}
"""
            )

            if result.coefficients is not None and not result.coefficients.empty:
                st.plotly_chart(viz.coefficient_figure(result), width="stretch")

            with st.expander("⚠️ Known limitations of this model"):
                for limitation in LIMITATIONS:
                    st.markdown(f"- {limitation}")

            st.download_button(
                "⬇️ Download the forecast as CSV",
                pd.DataFrame(
                    {
                        "Date": result.future_index,
                        "Forecast close": result.future_close.values,
                        "Lower band": result.lower_band.values,
                        "Upper band": result.upper_band.values,
                    }
                ).to_csv(index=False).encode("utf-8"),
                file_name=f"stocklens_forecast_{forecast_selector}.csv",
                mime="text/csv",
            )

# -------------------------------- GLOSSARY --------------------------------- #
with tab_glossary:
    st.subheader("📖 Jargon Translator")
    st.markdown(
        "Every technical term used anywhere in this app, explained without "
        "jargon. Use the search box to find a term quickly."
    )

    frame = glossary.glossary_dataframe()
    search = st.text_input("Search the glossary", value="", placeholder="e.g. RSI, drawdown, MAE")
    if search:
        mask = frame.apply(
            lambda row: search.lower() in " ".join(str(value).lower() for value in row.values), axis=1
        )
        frame = frame[mask]

    # The "id" column is the internal key used by the code, not something a
    # beginner needs, so it is dropped from the table shown here.
    st.dataframe(frame.drop(columns=["id"]), width="stretch", hide_index=True)

    st.markdown("### Methodology, step by step")
    st.markdown(
        f"""
1. **Data collection.** Daily open/high/low/close prices are downloaded from
   Yahoo Finance's public chart service for each ticker and for the benchmark.
   Every file is validated (duplicates removed, missing closes dropped, obvious
   data errors flagged) and cached locally for {config.CACHE_TTL_HOURS} hours.
   If the download fails, the app falls back to a local CSV in `data/user/`, and
   finally to clearly-labelled synthetic sample data. {glossary.explain('data_source')}

2. **Indicator calculation.**
   - {glossary.explain('sma20')} (window {config.SMA_FAST_WINDOW})
   - {glossary.explain('sma50')} (window {config.SMA_SLOW_WINDOW})
   - {glossary.explain('rsi')} (window {config.RSI_WINDOW})
   - {glossary.explain('volatility')}
   - {glossary.explain('max_drawdown')}

3. **Performance comparison.** Each ticker's total return is compared with the
   benchmark's over exactly the same days, and all series are rebased to 100
   so they can be drawn on one axis. {glossary.explain('rebased')}

4. **Suitability assessment.** Fixed thresholds produce a descriptive category
   and the numbers behind it are shown next to the label. There is no buy/sell
   output anywhere in the code. {glossary.explain('no_advice')}

5. **Forecasting.** A ridge regression on lagged returns and technical features
   predicts the next day's return. It is trained on the older {config.TRAIN_FRACTION:.0%}
   of the history and tested on the most recent {1 - config.TRAIN_FRACTION:.0%}, which is
   never shuffled. Accuracy is reported as MAE, RMSE and direction accuracy,
   against a naive baseline. {glossary.explain('model_limitations')}
"""
    )

    st.markdown("### Reproducibility notes")
    st.markdown(
        f"""
- All random or time-dependent behaviour is fixed: the synthetic sample data uses a
  fixed seed, and the model is a deterministic ridge regression.
- Results change every day simply because new market data arrives.
- The forecast band uses a {config.CONFIDENCE_LEVEL:.0%} confidence interval built from the
  model's own past errors - it is a guide to scale, not a promise.
- Source code layout: `data_loader.py`, `indicators.py`, `performance.py`,
  `suitability.py`, `forecaster.py`, `viz.py`, `glossary.py`, `config.py`, `app.py`.
"""
    )
