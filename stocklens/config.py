"""Central configuration for the StockLens application.

Everything that a user (or a grader) might reasonably want to tweak lives here:
default tickers, the benchmark, the meaning of each timeframe, every threshold
used by the suitability rules, and the file-system layout.

Keeping these values in one module means the rules in :mod:`stocklens.suitability`
stay transparent - a reader can see every cut-off in a single screen - and it
means the dashboard never hard-codes a magic number.

All monetary values are percentages unless the name says otherwise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

# --------------------------------------------------------------------------- #
# Branding / user-facing text
# --------------------------------------------------------------------------- #

APP_TITLE: Final[str] = "StockLens"
APP_SUBTITLE: Final[str] = (
    "A beginner-friendly stock tracker, performance comparator, risk analyzer "
    "and short-term forecaster."
)

DISCLAIMER_SHORT: Final[str] = (
    "Educational tool only - not financial or investment advice."
)

DISCLAIMER_LONG: Final[str] = (
    "**Educational use only.** StockLens was built for a university assignment to "
    "demonstrate data analysis, risk measurement and forecasting technique. It is "
    "**not** financial, investment, tax or legal advice, and nothing here is a "
    "recommendation to buy, sell or hold any security.\n\n"
    "* Forecasts are **probabilistic estimates**, not guarantees. A model that has "
    "been accurate in the past can still be wrong tomorrow, because markets are "
    "influenced by news, sentiment and events that no historical pattern contains.\n"
    "* Past performance does not predict future results.\n"
    "* Data is downloaded from a third party and may be delayed, incomplete or "
    "wrong. Always verify figures against a trusted source before using them.\n\n"
    "You are responsible for any decisions you make."
)

# --------------------------------------------------------------------------- #
# Data defaults
# --------------------------------------------------------------------------- #

#: Tickers pre-selected in the dashboard. Mixed on purpose so the comparison
#: screen shows something interesting straight away (a growth stock, a software
#: giant, a broad-market ETF and a defensive consumer staple).
DEFAULT_TICKERS: Final[tuple[str, ...]] = ("AAPL", "MSFT", "SPY", "KO")

#: Default benchmark index: the S&P 500 (Yahoo Finance symbol ``^GSPC``).
DEFAULT_BENCHMARK: Final[str] = "^GSPC"

#: How much calendar history to request per trading day we actually need.
#: A factor above 1.0 protects against weekends, holidays and exchange closures.
CALENDAR_DAYS_PER_TRADING_DAY: Final[float] = 1.6

#: Minimum number of price rows required before we trust the statistics.
#: Below this the app shows a "not enough data" warning instead of numbers.
#: 21 rows is one month of trading, the shortest window the dashboard offers.
MIN_HISTORY_ROWS: Final[int] = 21

#: Below this many rows the risk numbers are flagged as a rough estimate.
RELIABLE_HISTORY_ROWS: Final[int] = 60

#: Number of trading days per year, used to annualise volatility and returns.
TRADING_DAYS_PER_YEAR: Final[int] = 252

# --------------------------------------------------------------------------- #
# Timeframes (measured in *trading days*, which is how markets actually move)
# --------------------------------------------------------------------------- #

TIMEFRAME_TRADING_DAYS: Final[dict[str, int]] = {
    "1 Month": 21,
    "3 Months": 63,
    "6 Months": 126,
    "1 Year": 252,
    "3 Years": 756,
}

DEFAULT_TIMEFRAME: Final[str] = "1 Year"

# --------------------------------------------------------------------------- #
# Indicator settings
# --------------------------------------------------------------------------- #

SMA_FAST_WINDOW: Final[int] = 20  # "short-term trend"
SMA_SLOW_WINDOW: Final[int] = 50  # "medium-term trend"
RSI_WINDOW: Final[int] = 14  # standard momentum oscillator
VOLATILITY_WINDOW: Final[int] = 20  # rolling window for the volatility estimate
DRAWDOWN_WINDOW: Final[int] = 252  # window used for the drawdown chart

#: Extra rows of history fetched so that the 50-day average is available on the
#: very first day of the selected window (otherwise the first 49 days are blank).
WARMUP_TRADING_DAYS: Final[int] = 60

# --------------------------------------------------------------------------- #
# Suitability / risk-rule thresholds
#
# Every threshold below is referenced by a rule in :mod:`stocklens.suitability`
# and is shown to the user in the "Why?" column, so nothing is a black box.
# --------------------------------------------------------------------------- #

#: Annualised volatility bands (percent).
VOL_LOW_PCT: Final[float] = 25.0  # calmer than the typical large-cap stock
VOL_MEDIUM_PCT: Final[float] = 40.0  # normal-to-elevated
VOL_HIGH_PCT: Final[float] = 55.0  # genuinely jumpy

#: RSI bands (the classic 30 / 70 cut-offs).
RSI_OVERSOLD: Final[float] = 30.0
RSI_OVERBOUGHT: Final[float] = 70.0

#: Return bands over the selected window (percent).
RETURN_MOMENTUM_PCT: Final[float] = 25.0  # strong, clear upward move
RETURN_STEADY_PCT: Final[float] = 8.0  # modest move in either direction

#: Maximum-drawdown bands (percent, negative numbers).
DRAWDOWN_CAUTION_PCT: Final[float] = -25.0
DRAWDOWN_SEVERE_PCT: Final[float] = -40.0

#: How far above its 50-day average the price must be to count as a strong trend.
TREND_STRONG_PCT: Final[float] = 5.0

# --------------------------------------------------------------------------- #
# Forecasting settings
# --------------------------------------------------------------------------- #

#: How many days ahead the model projects.
FORECAST_HORIZON_DAYS: Final[int] = 5

#: Fraction of the history used for training. The rest is a strict
#: chronological "future" test set - never shuffled.
TRAIN_FRACTION: Final[float] = 0.8

#: Ridge regularisation strength. Larger = simpler, less over-fitted model.
RIDGE_ALPHA: Final[float] = 1.0

#: Number of past daily returns used as predictors.
N_LAGS: Final[int] = 5

#: Confidence level of the prediction band.
CONFIDENCE_LEVEL: Final[float] = 0.95

#: Minimum rows needed before a forecast is attempted at all.
MIN_ROWS_FOR_FORECAST: Final[int] = 120

# --------------------------------------------------------------------------- #
# File-system layout
# --------------------------------------------------------------------------- #

PACKAGE_DIR: Final[Path] = Path(__file__).resolve().parent
BASE_DIR: Final[Path] = PACKAGE_DIR.parent
DATA_DIR: Final[Path] = BASE_DIR / "data"
CACHE_DIR: Final[Path] = DATA_DIR / "cache"
SAMPLE_DIR: Final[Path] = DATA_DIR / "sample"

#: How long a cached download stays fresh before we re-download it.
CACHE_TTL_HOURS: Final[int] = 6

# --------------------------------------------------------------------------- #
# Visual palette (colour-blind friendly, consistent across every chart)
# --------------------------------------------------------------------------- #

COLOR_PRIMARY: Final[str] = "#2F6F9F"
COLOR_BENCHMARK: Final[str] = "#44474A"
COLOR_POSITIVE: Final[str] = "#2E8B57"
COLOR_NEGATIVE: Final[str] = "#C0392B"
COLOR_NEUTRAL: Final[str] = "#7F8C8D"
COLOR_GRID: Final[str] = "#E4E6E7"

#: Fixed colour per ticker so a company keeps the same colour on every chart.
TICKER_COLORS: Final[tuple[str, ...]] = (
    "#2F6F9F",
    "#E67E22",
    "#2E8B57",
    "#8E44AD",
    "#C0392B",
    "#16A085",
    "#D35400",
    "#2980B9",
    "#7D3C98",
    "#27AE60",
)

# --------------------------------------------------------------------------- #
# Suitability category metadata
# --------------------------------------------------------------------------- #

#: Human-readable summary and colour for each rule-based category.
SUITABILITY_CATEGORIES: Final[dict[str, dict[str, str]]] = {
    "Cautious / Steady": {
        "emoji": "🛡️",
        "colour": "#2E8B57",
        "summary": (
            "Historically steadier than most: smaller ups and downs, a shallow worst "
            "case. Usually the gentler place for a beginner to start learning."
        ),
    },
    "Balanced / Typical": {
        "emoji": "⚖️",
        "colour": "#2F6F9F",
        "summary": (
            "A fairly typical large-company profile: ordinary levels of movement and "
            "ordinary risk. Fine to study, but expect a bumpy ride."
        ),
    },
    "Momentum / Watchful": {
        "emoji": "🚀",
        "colour": "#E67E22",
        "summary": (
            "Strong recent upward trend, which often comes with higher day-to-day "
            "movement and a longer list of things that could pull it back."
        ),
    },
    "High-Volatility Caution": {
        "emoji": "⚠️",
        "colour": "#C0392B",
        "summary": (
            "Moves a lot in both directions and has suffered large declines before. "
            "A beginner should understand the risk before choosing this."
        ),
    },
    "Not Enough Data": {
        "emoji": "❓",
        "colour": "#7F8C8D",
        "summary": (
            "There is not enough price history (or it is too incomplete) to measure "
            "these characteristics responsibly."
        ),
    },
}
