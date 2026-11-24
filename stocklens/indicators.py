"""Technical indicators and risk statistics.

Every function here is written to be read: the maths is in the docstring, the
window length is a named argument with a sensible default, and nothing depends
on hidden global state.

The five measures a beginner most often meets in the wild are all covered:
simple moving averages, RSI, annualised volatility, maximum drawdown and
simple/annualised returns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config


# --------------------------------------------------------------------------- #
# Moving averages
# --------------------------------------------------------------------------- #


def simple_moving_average(series: pd.Series, window: int = config.SMA_FAST_WINDOW) -> pd.Series:
    """Compute a Simple Moving Average (SMA).

    An SMA is simply the average of the last ``window`` values, recomputed
    every day and drawn as a line. It smooths out daily noise so the direction
    of travel becomes visible.

    Args:
        series: Price series indexed by date.
        window: Number of periods to average over.

    Returns:
        A Series of the same index with NaN for the first ``window - 1`` rows
        (there is not enough history to average yet).
    """
    if window < 1:
        raise ValueError("window must be at least 1")
    return series.rolling(window=window, min_periods=window).mean()


# --------------------------------------------------------------------------- #
# Momentum
# --------------------------------------------------------------------------- #


def rsi(series: pd.Series, window: int = config.RSI_WINDOW) -> pd.Series:
    """Compute the Relative Strength Index (RSI).

    RSI answers one question: over the recent past, how much bigger have the
    up-days been than the down-days? It is expressed from 0 to 100.

    * Above 70 is traditionally read as "stretched" after a strong run.
    * Below 30 is traditionally read as "beaten down".

    We use Wilder's smoothing (an exponential moving average of average gains
    and losses) because that is the standard method the original RSI
    description uses.

    Args:
        series: Price series indexed by date.
        window: Look-back length in days (14 by convention).

    Returns:
        A Series of RSI values on the same index.
    """
    if window < 2:
        raise ValueError("window must be at least 2 for RSI")

    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()

    # If there were no losses at all the RSI is 100 by definition; guard the
    # division so we never produce inf or NaN.
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    values = 100.0 - (100.0 / (1.0 + rs))
    values = values.where(avg_loss.ne(0.0) | avg_gain.eq(0.0), 100.0)
    values = values.fillna(50.0)
    values[avg_gain.eq(0.0) & avg_loss.gt(0.0)] = 0.0
    values.name = "RSI"
    return values


# --------------------------------------------------------------------------- #
# Returns
# --------------------------------------------------------------------------- #


def daily_returns(series: pd.Series) -> pd.Series:
    """Compute day-over-day percentage returns.

    Returns:
        A Series of simple returns (0.01 means +1%).
    """
    return series.pct_change().dropna()


def total_return_pct(series: pd.Series) -> float:
    """Total percentage return over the whole series.

    Returns:
        ``(last / first - 1) * 100``.
    """
    clean = series.dropna()
    if len(clean) < 2 or clean.iloc[0] == 0:
        return float("nan")
    return float((clean.iloc[-1] / clean.iloc[0] - 1.0) * 100.0)


def annualised_return_pct(series: pd.Series) -> float:
    """Annualised percentage return (compound daily growth, scaled to a year).

    Args:
        series: Price series indexed by date.

    Returns:
        The annualised return in percent, or NaN if the series is too short.
    """
    clean = series.dropna()
    periods = len(clean) - 1
    if periods < 2 or clean.iloc[0] <= 0 or clean.iloc[-1] <= 0:
        return float("nan")
    total_growth = clean.iloc[-1] / clean.iloc[0]
    daily_rate = total_growth ** (1.0 / periods) - 1.0
    return float(daily_rate * config.TRADING_DAYS_PER_YEAR * 100.0)


def cagr_pct(series: pd.Series) -> float:
    """Compound Annual Growth Rate based on calendar time rather than day count."""
    clean = series.dropna()
    if len(clean) < 2 or clean.iloc[0] <= 0:
        return float("nan")
    years = (clean.index[-1] - clean.index[0]).days / 365.25
    if years <= 0:
        return float("nan")
    return float(((clean.iloc[-1] / clean.iloc[0]) ** (1.0 / years) - 1.0) * 100.0)


# --------------------------------------------------------------------------- #
# Risk
# --------------------------------------------------------------------------- #


def annualised_volatility_pct(returns: pd.Series) -> float:
    """Annualised volatility of a daily return series, in percent.

    Volatility is the standard deviation of daily returns, scaled up to a year
    by multiplying by the square root of the number of trading days (252).
    The result is the "typical size of a daily surprise" in annual terms.

    Args:
        returns: Daily percentage returns as decimals (0.01 = 1%).

    Returns:
        Annualised volatility in percent, or NaN if there is not enough data.
    """
    clean = returns.dropna()
    if len(clean) < 2:
        return float("nan")
    daily_std = float(clean.std(ddof=1))
    return float(daily_std * np.sqrt(config.TRADING_DAYS_PER_YEAR) * 100.0)


@dataclass
class DrawdownResult:
    """The outcome of a maximum-drawdown calculation.

    Attributes:
        max_drawdown_pct: Worst peak-to-trough fall, as a negative percent.
        peak_date: Date of the peak price before the worst fall.
        trough_date: Date of the lowest point after that peak.
        peak_price: Price at the peak.
        trough_price: Price at the trough.
    """

    max_drawdown_pct: float
    peak_date: pd.Timestamp | None
    trough_date: pd.Timestamp | None
    peak_price: float
    trough_price: float

    @property
    def recovery_text(self) -> str:
        """Short plain-English description of how deep and how long the fall was."""
        if self.max_drawdown_pct != self.max_drawdown_pct:  # NaN check
            return "Not enough data to measure a drawdown."
        depth = abs(self.max_drawdown_pct)
        length = ""
        if self.peak_date is not None and self.trough_date is not None:
            days = (self.trough_date - self.peak_date).days
            length = f" lasting about {days} days"
        return f"The worst peak-to-trough fall was {depth:.1f}%{length}."


def drawdown_series(series: pd.Series) -> pd.Series:
    """Percentage fall from the running peak, day by day (0% at every new high)."""
    running_peak = series.cummax()
    return (series / running_peak - 1.0) * 100.0


def maximum_drawdown(series: pd.Series) -> DrawdownResult:
    """Find the largest peak-to-trough decline in a price series.

    This is the "how bad did it get, and how long was I underwater?" number.
    For every point in time we compare the price with the highest price seen
    so far, then keep the worst of those gaps.

    Args:
        series: Price series indexed by date.

    Returns:
        A :class:`DrawdownResult` describing the worst decline.
    """
    clean = series.dropna()
    if len(clean) < 2:
        return DrawdownResult(float("nan"), None, None, float("nan"), float("nan"))

    drawdowns = drawdown_series(clean)
    trough_date = drawdowns.idxmin()
    worst = float(drawdowns.loc[trough_date])
    history = clean.loc[:trough_date]
    peak_date = history.idxmax()
    return DrawdownResult(
        max_drawdown_pct=worst,
        peak_date=peak_date,
        trough_date=trough_date,
        peak_price=float(clean.loc[peak_date]),
        trough_price=float(clean.loc[trough_date]),
    )


def worst_day_pct(returns: pd.Series) -> float:
    """Largest single-day loss, in percent (a negative number)."""
    clean = returns.dropna()
    if len(clean) < 1:
        return float("nan")
    return float(clean.min() * 100.0)


# --------------------------------------------------------------------------- #
# Combined panel
# --------------------------------------------------------------------------- #


def indicator_panel(series: pd.Series) -> pd.DataFrame:
    """Build a table of price, moving averages, RSI, volatility and drawdown.

    Args:
        series: Price series indexed by date.

    Returns:
        A DataFrame with the columns ``Close``, ``SMA20``, ``SMA50``, ``RSI``,
        ``RollingVol`` (annualised, 20-day window) and ``Drawdown``.
    """
    frame = pd.DataFrame({"Close": series})
    frame["SMA20"] = simple_moving_average(series, config.SMA_FAST_WINDOW)
    frame["SMA50"] = simple_moving_average(series, config.SMA_SLOW_WINDOW)
    frame["RSI"] = rsi(series, config.RSI_WINDOW)

    returns = series.pct_change()
    rolling_std = returns.rolling(window=config.VOLATILITY_WINDOW, min_periods=10).std(ddof=1)
    frame["RollingVol"] = rolling_std * np.sqrt(config.TRADING_DAYS_PER_YEAR) * 100.0
    frame["Drawdown"] = drawdown_series(series)
    frame["Return1D"] = returns * 100.0
    return frame


def describe_trend(price: float, sma_fast: float, sma_slow: float) -> str:
    """Turn the relationship between price and its averages into one sentence.

    Args:
        price: Latest close.
        sma_fast: Latest 20-day average.
        sma_slow: Latest 50-day average.

    Returns:
        A plain-English description of the trend, for example
        ``"price above both its 20-day and 50-day averages (upward trend)"``.
    """
    if any(value != value for value in (price, sma_fast, sma_slow)):
        return "Trend not available yet - not enough history for the moving averages."

    above_fast = price > sma_fast
    above_slow = price > sma_slow
    fast_above_slow = sma_fast > sma_slow

    if above_fast and above_slow and fast_above_slow:
        return "price is above both its 20-day and 50-day averages, and the 20-day average is above the 50-day average (clear upward trend)"
    if above_fast and above_slow:
        return "price is above both its 20-day and 50-day averages (upward trend)"
    if above_fast and not above_slow:
        return "price is above its 20-day average but below its 50-day average (short-term bounce inside a weaker longer-term trend)"
    if not above_fast and above_slow:
        return "price is below its 20-day average but still above its 50-day average (short-term cooling within an upward trend)"
    if not above_fast and not above_slow and not fast_above_slow:
        return "price is below both its 20-day and 50-day averages, and the 20-day average is below the 50-day average (clear downward trend)"
    return "price is below both its 20-day and 50-day averages (downward trend)"
