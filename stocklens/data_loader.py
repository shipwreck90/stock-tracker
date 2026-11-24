"""Downloading, validating and caching market data.

The module is deliberately defensive: a university demo should never die because
one ticker is misspelled or the lecture room Wi-Fi has dropped. Every symbol is
fetched through a chain of fallbacks:

1. Yahoo Finance's public chart service (HTTP, no API key needed).
2. The ``yfinance`` package, if it happens to be installed and working.
3. A local CSV file (yours, or a bundled sample) from the ``data/`` folder.

Whatever route is used, the result is validated, cached on disk and reported
back to the caller so the dashboard can show honest warnings.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Final, Iterable, Sequence

import numpy as np
import pandas as pd

from . import config

# --------------------------------------------------------------------------- #
# HTTP fetching
# --------------------------------------------------------------------------- #

_YAHOO_HOSTS: Final[tuple[str, ...]] = (
    "https://query1.finance.yahoo.com",
    "https://query2.finance.yahoo.com",
)

_UA: Final[str] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


class DataSourceError(RuntimeError):
    """Raised when a symbol could not be obtained from any source."""


def _fetch_via_http(symbol: str, start: date, end: date) -> pd.DataFrame:
    """Fetch daily OHLCV data from Yahoo Finance's public chart endpoint.

    Args:
        symbol: Ticker symbol, for example ``"AAPL"`` or ``"^GSPC"``.
        start: First calendar date to include.
        end: Last calendar date to include (inclusive).

    Returns:
        A DataFrame indexed by ``Date`` with columns
        ``Open, High, Low, Close, Adj Close, Volume``.

    Raises:
        DataSourceError: If every host fails or returns unusable data.
    """
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - requests is a hard dependency
        raise DataSourceError("The 'requests' package is not installed.") from exc

    period1 = int(datetime.combine(start, datetime.min.time()).timestamp())
    period2 = int(datetime.combine(end + timedelta(days=1), datetime.min.time()).timestamp())
    params = {
        "period1": period1,
        "period2": period2,
        "interval": "1d",
        "events": "div,splits,capitalGains",
        "includeAdjustedClose": "true",
    }
    headers = {"User-Agent": _UA, "Accept": "application/json"}

    last_error: Exception | None = None
    for host in _YAHOO_HOSTS:
        url = f"{host}/v8/finance/chart/{symbol}"
        try:
            response = requests.get(url, params=params, headers=headers, timeout=20)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:  # noqa: BLE001 - we want to try the next host
            last_error = exc
            continue

        chart = (payload or {}).get("chart") or {}
        if chart.get("error"):
            last_error = DataSourceError(str(chart["error"]))
            continue
        results = chart.get("result") or []
        if not results:
            last_error = DataSourceError("Empty result set returned.")
            continue

        result = results[0]
        timestamps = result.get("timestamp") or []
        quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
        adj_block = ((result.get("indicators") or {}).get("adjclose") or [{}])
        adj_close = (adj_block[0] if adj_block else {}).get("adjclose")

        if not timestamps:
            last_error = DataSourceError("No timestamps returned.")
            continue

        frame = pd.DataFrame(
            {
                "Open": quote.get("open"),
                "High": quote.get("high"),
                "Low": quote.get("low"),
                "Close": quote.get("close"),
                "Adj Close": adj_close if adj_close is not None else quote.get("close"),
                "Volume": quote.get("volume"),
            },
            index=pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None).normalize(),
        )
        frame.index.name = "Date"
        return frame

    raise DataSourceError(f"HTTP fetch failed for {symbol}: {last_error}")


def _fetch_via_yfinance(symbol: str, start: date, end: date) -> pd.DataFrame:
    """Fetch daily OHLCV data using the optional ``yfinance`` package.

    Args:
        symbol: Ticker symbol to download.
        start: First calendar date to include.
        end: Last calendar date to include (inclusive).

    Returns:
        A DataFrame indexed by ``Date`` with the same columns as
        :func:`_fetch_via_http`.

    Raises:
        DataSourceError: If ``yfinance`` is unavailable or returns nothing.
    """
    try:
        import yfinance  # type: ignore[import-not-found]
    except ImportError as exc:
        raise DataSourceError("yfinance is not installed.") from exc

    handle = yfinance.Ticker(symbol)
    raw = handle.history(start=start, end=end + timedelta(days=1), auto_adjust=False)
    if raw is None or raw.empty:
        raise DataSourceError("yfinance returned no rows.")

    frame = raw.copy()
    frame.columns = [str(c) for c in frame.columns]
    if "Close" in frame.columns:
        frame["Adj Close"] = frame.get("Adj Close", frame["Close"])
    frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
    frame.index.name = "Date"
    ordered = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    return frame.reindex(columns=[c for c in ordered if c in frame.columns])


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def validate_frame(frame: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, list[str]]:
    """Check a downloaded price frame and clean it up.

    The checks are deliberately the kind a careful analyst would make by hand:
    no duplicate days, no missing days silently kept, no zero prices, no
    obviously broken percentage changes.

    Args:
        frame: Raw price frame indexed by date.
        symbol: Ticker the frame belongs to, used in warning messages.

    Returns:
        A tuple of ``(cleaned_frame, warnings)``.
    """
    warnings: list[str] = []
    if frame is None or frame.empty:
        raise DataSourceError(f"No data returned for {symbol}.")

    cleaned = frame.copy()
    cleaned = cleaned[~cleaned.index.duplicated(keep="last")].sort_index()
    cleaned = cleaned.dropna(subset=["Close"])
    cleaned = cleaned[cleaned["Close"] > 0]

    missing_columns = [c for c in ("Open", "High", "Low", "Close", "Volume") if c not in cleaned]
    if missing_columns:
        raise DataSourceError(f"Data for {symbol} is missing columns: {missing_columns}")

    if "Adj Close" not in cleaned.columns or cleaned["Adj Close"].isna().all():
        cleaned["Adj Close"] = cleaned["Close"]
        warnings.append(f"{symbol}: adjusted close was unavailable, using plain close.")

    # A single-day return above ~80% is almost always a data error (bad split
    # adjustment, missing decimal point) rather than a real market move.
    returns = cleaned["Adj Close"].pct_change().abs()
    if len(returns) > 1 and float(returns.max(skipna=True) or 0) > 0.8:
        warnings.append(
            f"{symbol}: a one-day move of {returns.max():.0%} was detected and "
            "is likely a data error rather than a real price change."
        )

    if len(cleaned) < config.MIN_HISTORY_ROWS:
        warnings.append(
            f"{symbol}: only {len(cleaned)} days of history available - "
            f"fewer than the {config.MIN_HISTORY_ROWS} needed for reliable statistics."
        )

    return cleaned, warnings


# --------------------------------------------------------------------------- #
# Caching
# --------------------------------------------------------------------------- #


def _safe_name(symbol: str) -> str:
    """Turn a ticker into a filesystem-safe file name stem."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", symbol)


def _cache_paths(symbol: str) -> tuple[Path, Path]:
    """Return the ``(data, meta)`` cache paths for a symbol."""
    stem = _safe_name(symbol)
    return config.CACHE_DIR / f"{stem}.csv", config.CACHE_DIR / f"{stem}.meta.json"


def _read_cache(
    symbol: str,
    max_age_hours: int,
    min_start: pd.Timestamp | None = None,
) -> pd.DataFrame | None:
    """Read a cached price frame if it exists and is fresh enough.

    Args:
        symbol: Ticker symbol.
        max_age_hours: Maximum age of the cache in hours.
        min_start: Earliest date the caller needs. A cache that does not reach
            this far back is rejected so a longer timeframe always downloads
            the extra history it needs.

    Returns:
        The cached DataFrame, or ``None`` if missing, stale or too short.
    """
    data_path, meta_path = _cache_paths(symbol)
    if not data_path.exists() or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        age_hours = (time.time() - float(meta.get("fetched_at_epoch", 0))) / 3600.0
        if age_hours > max_age_hours:
            return None
        frame = pd.read_csv(data_path, index_col=0, parse_dates=True)
        frame.index = pd.to_datetime(frame.index)
        if frame.index.tz is not None:
            frame.index = frame.index.tz_localize(None)
        frame.index = frame.index.normalize()
        frame.index.name = "Date"
        if min_start is not None and len(frame) and frame.index.min() > min_start:
            # Cached history is too short for the requested timeframe.
            return None
        return frame
    except Exception:  # noqa: BLE001 - a broken cache should never break the app
        return None


def _write_cache(symbol: str, frame: pd.DataFrame, source: str) -> None:
    """Persist a price frame and its metadata to the local cache folder."""
    try:
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        data_path, meta_path = _cache_paths(symbol)
        frame.to_csv(data_path)
        meta_path.write_text(
            json.dumps(
                {
                    "symbol": symbol,
                    "source": source,
                    "fetched_at_epoch": time.time(),
                    "rows": int(len(frame)),
                    "first_date": str(frame.index.min().date()),
                    "last_date": str(frame.index.max().date()),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception:  # noqa: BLE001 - caching is best-effort only
        pass


def _load_local_csv(symbol: str) -> pd.DataFrame | None:
    """Look for a user-supplied or bundled sample CSV for a symbol.

    Search order: ``data/user/<symbol>.csv`` then ``data/sample/<symbol>.csv``.

    Args:
        symbol: Ticker symbol to look for.

    Returns:
        The loaded DataFrame, or ``None`` if no local file matched.
    """
    candidates = [
        config.DATA_DIR / "user" / f"{_safe_name(symbol)}.csv",
        config.SAMPLE_DIR / f"{_safe_name(symbol)}.csv",
    ]
    for path in candidates:
        if path.exists():
            try:
                frame = pd.read_csv(path, index_col=0, parse_dates=True)
                frame.index = pd.to_datetime(frame.index)
                if frame.index.tz is not None:
                    frame.index = frame.index.tz_localize(None)
                frame.index = frame.index.normalize()
                frame.index.name = "Date"
                # Accept either a tidy single-symbol file or one with a
                # 'Close'/'Adj Close' column.
                if "Close" not in frame.columns:
                    for candidate in ("Adj Close", "adj_close", "adjclose", "price", "Close "):
                        if candidate in frame.columns:
                            frame["Close"] = frame[candidate]
                            break
                return frame
            except Exception:  # noqa: BLE001 - try the next candidate
                continue
    return None


# --------------------------------------------------------------------------- #
# Public container
# --------------------------------------------------------------------------- #


@dataclass
class MarketData:
    """Everything the rest of the app needs to know about the downloaded data.

    Attributes:
        closes: Wide DataFrame of adjusted closes, one column per ticker plus
            the benchmark column.
        ohlcv: Raw per-symbol price frames keyed by symbol.
        benchmark: The benchmark symbol that was requested.
        sources: Human-readable description of where each symbol came from.
        synthetic_symbols: Symbols served from generated sample data. These
            are **not real prices** and the UI warns about them.
        warnings: Non-fatal messages worth showing the user.
        errors: Symbols that could not be loaded at all.
    """

    closes: pd.DataFrame
    ohlcv: dict[str, pd.DataFrame]
    benchmark: str
    sources: dict[str, str] = field(default_factory=dict)
    synthetic_symbols: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)
    start: date | None = None
    end: date | None = None

    @property
    def tickers(self) -> list[str]:
        """Ticker symbols available in :attr:`closes` (excluding the benchmark)."""
        return [c for c in self.closes.columns if c != self.benchmark]

    @property
    def benchmark_closes(self) -> pd.Series:
        """The benchmark price series."""
        if self.benchmark in self.closes.columns:
            return self.closes[self.benchmark].dropna()
        return pd.Series(dtype="float64")

    @property
    def is_degraded(self) -> bool:
        """True when any symbol had to be served from a fallback source."""
        return bool(self.synthetic_symbols) or bool(self.errors)


def _date_range_for(timeframe_days: int, warmup_days: int = 0) -> tuple[date, date]:
    """Convert a trading-day requirement into a calendar date range.

    Args:
        timeframe_days: Number of trading days the user wants to analyse.
        warmup_days: Extra trading days requested so indicator windows are
            already populated on the first visible day.

    Returns:
        ``(start_date, end_date)`` as ``datetime.date`` objects.
    """
    end = date.today()
    total_days = int((timeframe_days + warmup_days) * config.CALENDAR_DAYS_PER_TRADING_DAY)
    start = end - timedelta(days=max(total_days, 45))
    return start, end


def load_market_data(
    tickers: Sequence[str],
    benchmark: str,
    timeframe_days: int = 252,
    warmup_days: int = config.WARMUP_TRADING_DAYS,
    use_cache: bool = True,
    max_age_hours: int = config.CACHE_TTL_HOURS,
) -> MarketData:
    """Load prices for every requested symbol, with graceful fallbacks.

    Args:
        tickers: Symbols to load (for example ``["AAPL", "MSFT"]``).
        benchmark: Benchmark symbol to load alongside them.
        timeframe_days: Trading days of history the caller needs.
        warmup_days: Extra trading days for indicator warm-up.
        use_cache: Whether to read from / write to the local cache.
        max_age_hours: How old a cached download may be before re-fetching.

    Returns:
        A populated :class:`MarketData` object. Symbols that fail completely
        are recorded in ``MarketData.errors`` instead of raising.
    """
    from . import sample_data  # local import keeps module import order simple

    start, end = _date_range_for(timeframe_days, warmup_days)
    symbols: list[str] = []
    for symbol in [*tickers, benchmark]:
        cleaned = symbol.strip().upper()
        if cleaned and cleaned not in symbols:
            symbols.append(cleaned)

    frames: dict[str, pd.DataFrame] = {}
    sources: dict[str, str] = {}
    warnings: list[str] = []
    errors: dict[str, str] = {}
    synthetic: list[str] = []

    for symbol in symbols:
        frame: pd.DataFrame | None = None
        source = ""
        load_errors: list[str] = []
        min_start = pd.Timestamp(start)

        if use_cache:
            cached = _read_cache(symbol, max_age_hours, min_start)
            if cached is not None and not cached.empty:
                frame = cached
                source = "local cache"

        if frame is None:
            for loader, label in (
                (_fetch_via_http, "Yahoo Finance (HTTP)"),
                (_fetch_via_yfinance, "yfinance"),
            ):
                try:
                    frame = loader(symbol, start, end)
                    source = label
                    break
                except Exception as exc:  # noqa: BLE001 - try the next loader
                    load_errors.append(f"{label}: {exc}")

        if frame is None:
            local = _load_local_csv(symbol)
            if local is not None:
                frame = local
                source = "local CSV file"
            elif symbol in sample_data.SAMPLE_SYMBOLS:
                # Only symbols we know about get synthetic data. A typo must
                # be reported as an error rather than silently fabricated.
                frame = sample_data.generate_sample(symbol, start, end)
                source = "generated sample data (SYNTHETIC - not real prices)"
                synthetic.append(symbol)
            else:
                detail = load_errors[0] if load_errors else "no data returned"
                errors[symbol] = (
                    "No price history could be found for this symbol. It may be "
                    f"misspelled, delisted, or not covered by the data provider. "
                    f"({detail})"
                )
                continue

        try:
            cleaned, frame_warnings = validate_frame(frame, symbol)
        except DataSourceError as exc:
            errors[symbol] = str(exc)
            continue

        warnings.extend(frame_warnings)
        frames[symbol] = cleaned
        sources[symbol] = source
        if source == "local cache":
            # Refresh the cache timestamp so the freshness window slides forward.
            _write_cache(symbol, cleaned, "local cache")

    if not frames:
        raise DataSourceError(
            "No market data could be loaded for any symbol. Check your internet "
            "connection or place CSV files in the 'data/user' folder."
        )

    close_columns = {}
    for symbol, frame in frames.items():
        close_columns[symbol] = frame["Adj Close"]

    closes = pd.DataFrame(close_columns).sort_index()
    closes = closes.loc[closes.index.max() - pd.Timedelta(days=int(timeframe_days * 3.2)):]

    # The benchmark column is used for relative comparisons; drop it from the
    # ticker list when the user also selected it as a ticker.
    benchmark = symbols[-1] if symbols else benchmark

    return MarketData(
        closes=closes,
        ohlcv=frames,
        benchmark=benchmark,
        sources=sources,
        synthetic_symbols=synthetic,
        warnings=warnings,
        errors=errors,
        start=start,
        end=end,
    )
