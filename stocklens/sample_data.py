"""Deterministic synthetic price data for offline demonstrations.

These files are **not real market data**. They are produced by a deterministic
random walk so that the dashboard still works when there is no internet
connection (for example during a university demonstration in a lecture room).

Any symbol served this way is tagged as synthetic, the UI shows a prominent
warning, and the filenames carry a ``SYNTHETIC_`` prefix so they can never be
mistaken for a real download.

Run this module directly to (re)generate the bundled sample files::

    python -m stocklens.sample_data
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

#: Symbols generated offline when the network is unavailable.
SAMPLE_SYMBOLS: tuple[str, ...] = ("AAPL", "MSFT", "SPY", "KO", "^GSPC")

#: Annualised volatility used for each sample symbol.
_SAMPLE_VOL: dict[str, float] = {
    "AAPL": 0.30,
    "MSFT": 0.26,
    "SPY": 0.16,
    "KO": 0.14,
    "^GSPC": 0.16,
}

#: Approximate annual drift for each sample symbol.
_SAMPLE_DRIFT: dict[str, float] = {
    "AAPL": 0.15,
    "MSFT": 0.14,
    "SPY": 0.09,
    "KO": 0.05,
    "^GSPC": 0.09,
}


def _business_days(start: date, end: date) -> pd.DatetimeIndex:
    """Return the set of business days between two dates."""
    return pd.bdate_range(start=start, end=end)


def generate_sample(symbol: str, start: date, end: date, seed: int = 7) -> pd.DataFrame:
    """Generate a synthetic OHLCV history for a symbol.

    The series is a geometric random walk with a fixed seed, so the same
    symbol always produces the same history within a project run.

    Args:
        symbol: Ticker symbol to fabricate a history for.
        start: First date of the series.
        end: Last date of the series.
        seed: Random seed for reproducibility.

    Returns:
        A DataFrame indexed by ``Date`` with the standard OHLCV columns.
    """
    index = _business_days(start, end)
    if len(index) == 0:
        index = _business_days(end - timedelta(days=400), end)

    vol = _SAMPLE_VOL.get(symbol, 0.25)
    drift = _SAMPLE_DRIFT.get(symbol, 0.08)
    rng = np.random.default_rng(abs(hash(symbol)) % (2**32) + seed)

    daily_vol = vol / np.sqrt(config.TRADING_DAYS_PER_YEAR)
    daily_drift = drift / config.TRADING_DAYS_PER_YEAR
    shocks = rng.normal(loc=daily_drift, scale=daily_vol, size=len(index))
    path = 100.0 * np.exp(np.cumsum(shocks))

    open_ = np.r_[path[0], path[:-1]]
    noise_high = np.abs(rng.normal(0, daily_vol / 2, size=len(index)))
    noise_low = np.abs(rng.normal(0, daily_vol / 2, size=len(index)))
    close = path
    high = np.maximum(open_, close) * (1 + noise_high)
    low = np.minimum(open_, close) * (1 - noise_low)
    volume = rng.integers(20_000_000, 90_000_000, size=len(index)).astype(float)

    frame = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Adj Close": close, "Volume": volume},
        index=index,
    )
    frame.index.name = "Date"
    return frame


def write_sample_files(directory: Path | None = None) -> list[Path]:
    """Write the bundled synthetic sample CSV files to disk.

    Args:
        directory: Destination folder. Defaults to ``data/sample``.

    Returns:
        The list of files that were written.
    """
    target = directory or config.SAMPLE_DIR
    target.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    end = date.today()
    start = end - timedelta(days=int(900 * config.CALENDAR_DAYS_PER_TRADING_DAY))
    for symbol in SAMPLE_SYMBOLS:
        frame = generate_sample(symbol, start, end)
        safe = symbol.replace("^", "INDEX_")
        path = target / f"{safe}.csv"
        frame.to_csv(path)
        written.append(path)
    return written


if __name__ == "__main__":  # pragma: no cover - manual utility
    files = write_sample_files()
    print("Wrote synthetic sample files:")
    for path in files:
        print(f"  - {path}")
