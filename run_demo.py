"""Headless demo and self-check.

Runs the whole pipeline without opening a browser so you can verify the
project works (and capture output for a report) straight from the terminal::

    python run_demo.py
    python run_demo.py --tickers AAPL,MSFT,KO --benchmark ^GSPC --timeframe "6 Months"

The script also performs a few automated sanity checks, including a check that
the suitability engine never emits buy/sell style language.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stocklens import config, glossary, performance, suitability  # noqa: E402
from stocklens.data_loader import load_market_data  # noqa: E402
from stocklens.forecaster import forecast, forecast_explanation  # noqa: E402
from stocklens.indicators import indicator_panel  # noqa: E402

RULE = "-" * 78


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the demo."""
    parser = argparse.ArgumentParser(description="StockLens headless demo")
    parser.add_argument(
        "--tickers",
        default=",".join(config.DEFAULT_TICKERS),
        help="Comma-separated list of tickers.",
    )
    parser.add_argument("--benchmark", default=config.DEFAULT_BENCHMARK, help="Benchmark symbol.")
    parser.add_argument(
        "--timeframe",
        default=config.DEFAULT_TIMEFRAME,
        choices=list(config.TIMEFRAME_TRADING_DAYS.keys()),
        help="How far back to look.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the demo and print a readable report."""
    args = parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    timeframe_days = config.TIMEFRAME_TRADING_DAYS[args.timeframe]

    print(RULE)
    print(f"{config.APP_TITLE} - headless demo")
    print(f"{config.DISCLAIMER_SHORT}")
    print(RULE)
    print(f"Tickers     : {', '.join(tickers)}")
    print(f"Benchmark   : {args.benchmark}")
    print(f"Timeframe   : {args.timeframe} ({timeframe_days} trading days)")
    print(RULE)

    data = load_market_data(tickers=tickers, benchmark=args.benchmark, timeframe_days=timeframe_days)

    print("\nDATA SOURCES")
    for symbol, source in sorted(data.sources.items()):
        rows = len(data.ohlcv.get(symbol, []))
        print(f"  {symbol:<8} {source:<45} {rows:>4} rows")
    if data.errors:
        print("\n  Errors:")
        for symbol, reason in data.errors.items():
            print(f"    {symbol}: {reason}")
    if data.warnings:
        print("\n  Warnings:")
        for note in data.warnings:
            print(f"    - {note}")

    closes = performance.select_window(data.closes, timeframe_days, data.benchmark)
    table = performance.build_performance_table(closes, data.benchmark)

    print(f"\nPERFORMANCE LEADERBOARD ({args.timeframe})")
    print(f"  {'Rank':<5}{'Ticker':<9}{'Total %':>10}{'Annual %':>11}{'vs bench':>11}{'Vol %':>9}{'MaxDD %':>10}")
    for _, row in table.iterrows():
        marker = " *" if row["Is benchmark"] else ""
        print(
            f"  {int(row['Rank']):<5}{str(row['Ticker']) + marker:<9}"
            f"{row['Total return (%)']:>10.2f}{row['Annualised return (%/yr)']:>11.2f}"
            f"{row['Return vs benchmark (pp)']:>11.2f}{row['Volatility (%/yr)']:>9.2f}"
            f"{row['Max drawdown (%)']:>10.2f}"
        )
    print("  (* = benchmark)")

    print("\nIN PLAIN ENGLISH")
    for sentence in performance.summary_sentences(table, data.benchmark):
        print(f"  - {sentence}")

    print(f"\nBEGINNER SUITABILITY & RISK PROFILE")
    profiles = suitability.assess_many(
        {column: closes[column].dropna() for column in closes.columns if column != data.benchmark},
        {column: data.closes[column].dropna() for column in closes.columns if column != data.benchmark},
    )
    for ticker, profile in profiles.items():
        print(f"\n  {profile.emoji} {ticker}: {profile.category}")
        for reason in profile.reasons:
            print(f"      WHY  : {reason}")
        for note in profile.observations:
            print(f"      NOTE : {note}")

    print(f"\nSHORT-TERM FORECAST ({config.FORECAST_HORIZON_DAYS} trading days)")
    for ticker in profiles:
        result = forecast(data.closes[ticker].dropna(), ticker=ticker)
        if not result.is_valid:
            print(f"\n  {ticker}: forecast unavailable - {'; '.join(result.errors)}")
            continue
        print(f"\n  {ticker}")
        for sentence in forecast_explanation(result):
            print(f"      - {sentence}")
        if result.future_close is not None:
            for date_value, close_value, low, high in zip(
                result.future_index, result.future_close, result.lower_band, result.upper_band
            ):
                print(
                    f"      {date_value.date()}  forecast {close_value:>9.2f}"
                    f"   band {low:>8.2f} - {high:>8.2f}"
                )

    # ---------------- automated self-checks ---------------- #
    print(f"\n{RULE}\nSELF-CHECKS")
    checks: list[tuple[str, bool]] = []

    checks.append(("Data loaded for at least one ticker", len(table) > 0))
    checks.append(("Leaderboard contains only valid numbers", bool(table.empty or table["Total return (%)"].notna().all())))
    checks.append(("Rebased series all start near 100", bool(rebased_ok(rebased_values(closes)))))
    checks.append(("No buy/sell language in suitability output", no_advice_language(profiles)))
    checks.append(("Glossary covers every exported metric id", glossary_ok()))

    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")

    failures = [label for label, passed in checks if not passed]
    print(RULE)
    if failures:
        print(f"{len(failures)} check(s) failed: {', '.join(failures)}")
        return 1
    print("All checks passed.")
    return 0


def rebased_values(closes):
    """Return the rebased frame used by the self-check."""
    return performance.rebased_frame(closes)


def rebased_ok(frame) -> bool:
    """True when every rebased column starts close to 100."""
    if frame.empty:
        return True
    first_rows = frame.dropna().head(1)
    if first_rows.empty:
        return True
    return bool(((first_rows - 100.0).abs() < 1.0).all(axis=None))


def no_advice_language(profiles) -> bool:
    """True when no suitability message contains advice-like wording."""
    forbidden = ("buy", "sell", "should invest", "recommend")
    for profile in profiles.values():
        text = " ".join(profile.reasons + profile.observations + [profile.summary]).lower()
        if any(word in text for word in forbidden):
            print(f"      offending text for {profile.ticker}: {text}")
            return False
    return True


def glossary_ok() -> bool:
    """True when the glossary contains the metrics the app relies on."""
    required = {"total_return", "rsi", "volatility", "max_drawdown", "mae", "rmse", "prediction_interval"}
    return required.issubset(set(glossary.GLOSSARY.keys()))


if __name__ == "__main__":
    raise SystemExit(main())
