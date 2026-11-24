"""Plotly chart builders.

Keeping every chart in one module makes the dashboard code readable and means
the colour scheme, axis labels and plain-English annotations stay consistent
across every tab.

All functions return a ``plotly.graph_objects.Figure`` which Streamlit renders
directly.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from . import config
from .forecaster import ForecastResult


#: Explicit dark text colour. Streamlit's dark theme sets a light global font
#: colour, which would otherwise render Plotly's titles and axis labels almost
#: invisibly on the white chart background.
TEXT_COLOR: str = "#1F2A33"
TICK_COLOR: str = "#3D4852"


def _colour_for(index: int, total: int) -> str:
    """Pick a stable colour from the palette."""
    if total <= 1:
        return config.TICKER_COLORS[0]
    return config.TICKER_COLORS[index % len(config.TICKER_COLORS)]


def _base_layout(fig: go.Figure, title: str, y_title: str) -> go.Figure:
    """Apply the shared chart styling."""
    fig.update_layout(
        title={"text": title, "font": {"size": 17, "color": TEXT_COLOR}},
        template="plotly_white",
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
        hovermode="x unified",
        font={"color": TEXT_COLOR, "size": 13},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "font": {"color": TEXT_COLOR},
            "bgcolor": "rgba(255,255,255,0)",
        },
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
    )
    fig.update_xaxes(
        showgrid=False,
        tickfont={"color": TICK_COLOR},
        linecolor="#B0B3B6",
        title={"text": "", "font": {"color": TEXT_COLOR}},
    )
    fig.update_yaxes(
        title={"text": y_title, "font": {"color": TEXT_COLOR}},
        tickfont={"color": TICK_COLOR},
        gridcolor=config.COLOR_GRID,
        zerolinecolor=config.COLOR_GRID,
    )
    return fig


def rebased_performance_figure(rebased_frame: pd.DataFrame, benchmark: str) -> go.Figure:
    """Line chart of every ticker's performance, all starting at 100.

    Args:
        rebased_frame: Output of :func:`stocklens.performance.rebased_frame`.
        benchmark: Benchmark column, drawn in a neutral colour.

    Returns:
        A Plotly figure.
    """
    fig = go.Figure()
    columns = list(rebased_frame.columns)
    for index, column in enumerate(columns):
        is_benchmark = column == benchmark
        fig.add_trace(
            go.Scatter(
                x=rebased_frame.index,
                y=rebased_frame[column],
                name=str(column),
                mode="lines",
                line={
                    "width": 3 if is_benchmark else 2,
                    "color": config.COLOR_BENCHMARK if is_benchmark else _colour_for(index, len(columns)),
                    "dash": "dash" if is_benchmark else "solid",
                },
            )
        )
    fig.add_hline(y=100, line_dash="dot", line_color="#B0B3B6")
    fig.update_yaxes(tickformat=".0f")
    return _base_layout(
        fig,
        "Performance with every stock starting at 100 (higher line = better performance)",
        "Rebased value (start = 100)",
    )


def relative_performance_figure(relative_frame: pd.DataFrame, benchmark: str) -> go.Figure:
    """Chart showing who is ahead of or behind the benchmark."""
    fig = go.Figure()
    columns = list(relative_frame.columns)
    for index, column in enumerate(columns):
        fig.add_trace(
            go.Scatter(
                x=relative_frame.index,
                y=relative_frame[column],
                name=str(column),
                mode="lines",
                line={"width": 2, "color": _colour_for(index, len(columns))},
            )
        )
    fig.add_hline(y=100, line_dash="dot", line_color=config.COLOR_BENCHMARK)
    fig.update_yaxes(tickformat=".0f")
    return _base_layout(
        fig,
        "Relative to the benchmark (100 = exactly in line with the benchmark)",
        "Relative performance index",
    )


def returns_bar_figure(table: pd.DataFrame, benchmark: str) -> go.Figure:
    """Horizontal bar chart of total returns, benchmark highlighted separately."""
    if table.empty:
        return go.Figure()
    plotted = table[~table["Is benchmark"]].sort_values("Total return (%)")
    colors = [
        config.COLOR_POSITIVE if value >= 0 else config.COLOR_NEGATIVE
        for value in plotted["Total return (%)"]
    ]
    fig = go.Figure(
        go.Bar(
            x=plotted["Total return (%)"],
            y=plotted["Ticker"],
            orientation="h",
            marker={"color": colors},
            text=[f"{value:+.1f}%" for value in plotted["Total return (%)"]],
            textposition="outside",
            hovertemplate="%{y}: %{x:+.2f}%<extra></extra>",
        )
    )
    bench_row = table[table["Is benchmark"]]
    if not bench_row.empty:
        bench_value = float(bench_row["Total return (%)"].iloc[0])
        fig.add_hline(
            y=bench_value,
            line_dash="dash",
            line_color=config.COLOR_BENCHMARK,
            annotation_text=f"{benchmark}: {bench_value:+.1f}%",
            annotation_position="top left",
            annotation_font={"color": TEXT_COLOR},
        )
    fig.update_xaxes(tickformat=".0f")
    return _base_layout(fig, "Total return over the selected window", "Total return (%)")


def indicator_figure(panel: pd.DataFrame, ticker: str) -> go.Figure:
    """Chart of price with the 20-day and 50-day moving averages."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=panel.index, y=panel["Close"], name="Price", mode="lines", line={"width": 2.5, "color": config.COLOR_PRIMARY})
    )
    if "SMA20" in panel:
        fig.add_trace(
            go.Scatter(x=panel.index, y=panel["SMA20"], name="20-day average", mode="lines", line={"width": 1.8, "color": config.COLOR_POSITIVE})
        )
    if "SMA50" in panel:
        fig.add_trace(
            go.Scatter(x=panel.index, y=panel["SMA50"], name="50-day average", mode="lines", line={"width": 1.8, "color": config.COLOR_NEGATIVE})
        )
    return _base_layout(fig, f"{ticker}: price versus its moving averages", "Price")


def rsi_figure(panel: pd.DataFrame, ticker: str) -> go.Figure:
    """Chart of the RSI oscillator with the 30 / 70 guide lines."""
    fig = go.Figure()
    if "RSI" in panel:
        fig.add_trace(
            go.Scatter(x=panel.index, y=panel["RSI"], name="RSI (14-day)", mode="lines", line={"width": 2, "color": config.COLOR_PRIMARY})
        )
    fig.add_hline(y=config.RSI_OVERBOUGHT, line_dash="dash", line_color=config.COLOR_NEGATIVE,
                  annotation_text=f"overbought ({config.RSI_OVERBOUGHT:.0f})", annotation_position="top right",
                  annotation_font={"color": TEXT_COLOR})
    fig.add_hline(y=config.RSI_OVERSOLD, line_dash="dash", line_color=config.COLOR_POSITIVE,
                  annotation_text=f"oversold ({config.RSI_OVERSOLD:.0f})", annotation_position="bottom right",
                  annotation_font={"color": TEXT_COLOR})
    fig.add_hrect(y0=config.RSI_OVERSOLD, y1=config.RSI_OVERBOUGHT, fillcolor="#F4F6F7", opacity=0.6, line_width=0)
    fig.update_yaxes(range=[0, 100])
    return _base_layout(fig, f"{ticker}: momentum oscillator (RSI)", "RSI")


def drawdown_figure(panel: pd.DataFrame, ticker: str) -> go.Figure:
    """Chart of how far the price was below its previous peak."""
    fig = go.Figure()
    if "Drawdown" in panel:
        fig.add_trace(
            go.Scatter(
                x=panel.index,
                y=panel["Drawdown"],
                name="Drawdown from peak",
                mode="lines",
                line={"width": 2, "color": config.COLOR_NEGATIVE},
                fill="tozeroy",
                fillcolor="rgba(192, 57, 43, 0.15)",
            )
        )
    return _base_layout(fig, f"{ticker}: how far below its peak the price traded", "Drawdown (%)")


def forecast_figure(result: ForecastResult, history: pd.Series, ticker: str) -> go.Figure:
    """Chart of recent history plus the forecast and its uncertainty band."""
    fig = go.Figure()
    if history is not None and not history.empty:
        tail = history.tail(120)
        fig.add_trace(
            go.Scatter(x=tail.index, y=tail, name="Actual price", mode="lines", line={"width": 2.5, "color": config.COLOR_PRIMARY})
        )

    if result.is_valid and result.future_close is not None and result.future_index is not None:
        anchor_index = history.index[-1] if history is not None and not history.empty else None
        anchor_value = result.last_actual_close

        x_line = list(result.future_index)
        y_line = list(result.future_close)
        if anchor_index is not None:
            x_line = [anchor_index] + x_line
            y_line = [anchor_value] + y_line

        lower = list(result.lower_band) if result.lower_band is not None else []
        upper = list(result.upper_band) if result.upper_band is not None else []
        if anchor_index is not None and lower and upper:
            lower = [anchor_value] + lower
            upper = [anchor_value] + upper

        if lower and upper:
            fig.add_trace(
                go.Scatter(
                    x=x_line + x_line[::-1],
                    y=upper + lower[::-1],
                    fill="toself",
                    fillcolor="rgba(47, 111, 159, 0.16)",
                    line={"width": 0},
                    hoverinfo="skip",
                    name="Uncertainty band",
                )
            )

        fig.add_trace(
            go.Scatter(x=x_line, y=y_line, name="Model forecast", mode="lines+markers",
                       line={"width": 2.5, "color": config.COLOR_NEGATIVE, "dash": "dash"})
        )

    return _base_layout(
        fig,
        f"{ticker}: short-term forecast with an uncertainty band",
        "Closing price",
    )


def coefficient_figure(result: ForecastResult) -> go.Figure:
    """Bar chart of the fitted model coefficients (standardised units)."""
    if result.coefficients is None or result.coefficients.empty:
        return go.Figure()
    values = result.coefficients
    colors = [config.COLOR_POSITIVE if value >= 0 else config.COLOR_NEGATIVE for value in values]
    fig = go.Figure(
        go.Bar(
            x=values.values,
            y=values.index,
            orientation="h",
            marker={"color": colors},
            text=[f"{value:+.4f}" for value in values.values],
            textposition="outside",
        )
    )
    fig.add_vline(x=0, line_color="#B0B3B6")
    return _base_layout(
        fig,
        "What the model leans on (larger weight = stronger influence on the prediction)",
        "Standardised weight",
    )
