"""Side-by-side performance comparison against a benchmark.

This module answers the beginner's first question: *who is performing better,
and who is performing worse?* It produces

* a leaderboard table (total return, annualised return, return vs benchmark),
* rebased series so differently priced stocks can be compared on one chart,
* relative-return figures in percentage points, and
* plain-English summary sentences for the dashboard cards.
"""

from __future__ import annotations

import pandas as pd

from . import config
from .indicators import (
    annualised_return_pct,
    annualised_volatility_pct,
    daily_returns,
    maximum_drawdown,
    total_return_pct,
)


def rebased(series: pd.Series, base: float = 100.0) -> pd.Series:
    """Rebase a price series to a common starting value.

    Each value is divided by the first value in the window and multiplied by
    ``base``. This puts a 20-dollar stock and a 500-dollar stock on the same
    chart footing: whichever ends higher after the rebasing has performed
    better.

    Args:
        series: Price series indexed by date.
        base: Starting value of the rebased series (100 by convention).

    Returns:
        The rebased series.
    """
    clean = series.dropna()
    if clean.empty or clean.iloc[0] == 0:
        return pd.Series(dtype="float64")
    return (clean / clean.iloc[0]) * base


def rebased_frame(closes: pd.DataFrame, base: float = 100.0) -> pd.DataFrame:
    """Rebase every column of a wide price frame."""
    return closes.apply(lambda column: rebased(column, base))


def relative_return_points(closes: pd.DataFrame, benchmark: str) -> pd.Series:
    """Return each ticker's total return minus the benchmark's, in points.

    Args:
        closes: Wide frame of adjusted closes.
        benchmark: Column name of the benchmark.

    Returns:
        A Series of percentage-point differences (empty if the benchmark is
        missing).
    """
    if benchmark not in closes.columns:
        return pd.Series(dtype="float64")
    bench_return = total_return_pct(closes[benchmark])
    results = {c: total_return_pct(closes[c]) - bench_return for c in closes.columns}
    return pd.Series(results, dtype="float64").drop(index=benchmark, errors="ignore")


def build_performance_table(closes: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    """Build the leaderboard table for the selected window.

    Args:
        closes: Wide frame of adjusted closes (one column per ticker plus the
            benchmark column).
        benchmark: Column name of the benchmark series.

    Returns:
        A DataFrame indexed by ticker with the columns
        ``Start price``, ``End price``, ``Total return (%)``,
        ``Annualised return (%/yr)``, ``Return vs benchmark (pp)``,
        ``Volatility (%/yr)``, ``Max drawdown (%)`` and ``Rank``.
    """
    if closes.empty:
        return pd.DataFrame()

    bench_return = total_return_pct(closes[benchmark]) if benchmark in closes.columns else float("nan")
    rows: list[dict[str, object]] = []

    for column in closes.columns:
        series = closes[column].dropna()
        if series.empty:
            continue
        returns = daily_returns(series)
        drawdown = maximum_drawdown(series)
        rows.append(
            {
                "Ticker": column,
                "Start price": float(series.iloc[0]),
                "End price": float(series.iloc[-1]),
                "Total return (%)": total_return_pct(series),
                "Annualised return (%/yr)": annualised_return_pct(series),
                "Return vs benchmark (pp)": total_return_pct(series) - bench_return,
                "Volatility (%/yr)": annualised_volatility_pct(returns),
                "Max drawdown (%)": drawdown.max_drawdown_pct,
                "Is benchmark": column == benchmark,
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table = table.sort_values("Total return (%)", ascending=False).reset_index(drop=True)
    table["Rank"] = range(1, len(table) + 1)
    return table


def performance_leaders(table: pd.DataFrame) -> tuple[str | None, str | None]:
    """Return the best and worst performing tickers from a leaderboard.

    Args:
        table: Output of :func:`build_performance_table`.

    Returns:
        ``(best_ticker, worst_ticker)``; either may be ``None``.
    """
    if table.empty:
        return None, None
    ranked = table[~table["Is benchmark"]]
    if ranked.empty:
        return None, None
    best = ranked.iloc[0]["Ticker"]
    worst = ranked.iloc[-1]["Ticker"]
    return str(best), str(worst)


def summary_sentences(table: pd.DataFrame, benchmark: str) -> list[str]:
    """Write plain-English sentences describing the comparison.

    Args:
        table: Output of :func:`build_performance_table`.
        benchmark: Benchmark ticker, used in the wording.

    Returns:
        A list of sentences suitable for a dashboard summary card.
    """
    if table.empty:
        return ["No data available for the selected window."]

    sentences: list[str] = []
    ranked = table[~table["Is benchmark"]].sort_values("Total return (%)", ascending=False)
    if ranked.empty:
        return ["Only the benchmark was available for this selection."]

    best = ranked.iloc[0]
    worst = ranked.iloc[-1]
    sentences.append(
        f"**{best['Ticker']}** performed best over this window at "
        f"{best['Total return (%)']:+.1f}%, while **{worst['Ticker']}** performed "
        f"worst at {worst['Total return (%)']:+.1f}%."
    )

    beat = ranked[ranked["Return vs benchmark (pp)"] > 0]
    if len(beat) == len(ranked) and len(ranked) > 1:
        sentences.append(
            f"Every selected stock beat the {benchmark} benchmark - this was a "
            "strong period for the market overall."
        )
    elif len(beat) == 0:
        sentences.append(
            f"None of the selected stocks beat the {benchmark} benchmark over this window."
        )
    else:
        names = ", ".join(str(t) for t in beat["Ticker"])
        sentences.append(
            f"{names} beat the {benchmark} benchmark; the others fell short of it."
        )

    spread = float(ranked["Total return (%)"].max() - ranked["Total return (%)"].min())
    sentences.append(
        f"The gap between the best and worst performer was {spread:.1f} percentage points."
    )
    return sentences


def relative_performance_frame(closes: pd.DataFrame, benchmark: str, base: float = 100.0) -> pd.DataFrame:
    """Rebase every series relative to the benchmark's path.

    Args:
        closes: Wide frame of adjusted closes.
        benchmark: Column name of the benchmark.
        base: Starting value (100 by convention).

    Returns:
        A DataFrame where 100 means "exactly the same as the benchmark on that
        day", above 100 means "ahead of it" and below 100 means "behind it".
    """
    if benchmark not in closes.columns:
        return rebased_frame(closes, base)
    bench_rebased = rebased(closes[benchmark], base)
    result = pd.DataFrame(index=closes.index)
    for column in closes.columns:
        if column == benchmark:
            continue
        series = rebased(closes[column], base)
        ratio = series / bench_rebased.replace(0.0, pd.NA)
        result[column] = ratio.astype("float64") * base
    return result


def cumulative_return_curve(series: pd.Series) -> pd.Series:
    """Cumulative percentage return starting from zero."""
    returns = daily_returns(series)
    if returns.empty:
        return pd.Series(dtype="float64")
    return (1.0 + returns).cumprod() * 100.0 - 100.0


def timeframe_label(timeframe_days: int) -> str:
    """Turn a number of trading days into a friendly label."""
    mapping = {
        21: "1 month",
        63: "3 months",
        126: "6 months",
        252: "1 year",
        756: "3 years",
    }
    return mapping.get(int(timeframe_days), f"{int(timeframe_days)} trading days")


def timeframe_from_label(label: str) -> int:
    """Inverse of :func:`timeframe_label` for the built-in labels."""
    return config.TIMEFRAME_TRADING_DAYS.get(label, 252)


def select_window(
    closes: pd.DataFrame,
    timeframe_days: int,
    benchmark: str | None = None,
) -> pd.DataFrame:
    """Cut the price frame down to exactly the selected number of trading days.

    The reference series (the benchmark if present, otherwise the first
    column) supplies the date range, so every ticker is compared over exactly
    the same days even when a symbol is missing a price or two.

    Args:
        closes: Wide frame of adjusted closes.
        timeframe_days: Number of trading days the window should contain.
        benchmark: Column used as the date reference.

    Returns:
        The windowed frame.
    """
    if closes.empty:
        return closes

    reference = closes[benchmark] if benchmark in closes.columns else closes.iloc[:, 0]
    reference = reference.dropna()
    if len(reference) <= timeframe_days:
        return closes.copy()

    dates = reference.index[-timeframe_days:]
    start, end = dates[0], dates[-1]
    return closes.loc[(closes.index >= start) & (closes.index <= end)].copy()
