"""Rule-based "Beginner Suitability & Risk Profile" assessment.

This module classifies a stock into one of five descriptive categories using
**fixed, published thresholds** - never a black-box model - and returns the
exact numbers behind every label so a reader can disagree with it.

Important: this module produces *descriptions*, not instructions. There is no
"buy" or "sell" output anywhere, by design. A final safety check
(:func:`_assert_no_advice_language`) enforces that.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import config
from .indicators import (
    annualised_volatility_pct,
    annualised_return_pct,
    daily_returns,
    describe_trend,
    maximum_drawdown,
    rsi,
    simple_moving_average,
    total_return_pct,
    worst_day_pct,
)

#: Words that must never appear in a suitability message.
_FORBIDDEN_WORDS: tuple[str, ...] = ("buy", "sell", "should invest", "recommend", "must trade")


@dataclass
class SuitabilityResult:
    """The outcome of the rule engine for one ticker.

    Attributes:
        ticker: The symbol that was assessed.
        category: One of the keys of ``config.SUITABILITY_CATEGORIES``.
        summary: Plain-English description of the category.
        reasons: The triggered rules, each with the numbers that caused them.
        observations: Additional facts worth knowing (trend, RSI, worst day).
        metrics: The raw numbers used by the rules.
        emoji: Decorative emoji for the dashboard.
        colour: Hex colour used for the category badge.
    """

    ticker: str
    category: str
    summary: str
    reasons: list[str] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    emoji: str = ""
    colour: str = "#7F8C8D"

    @property
    def label(self) -> str:
        """Category name including its emoji, for display."""
        return f"{self.emoji} {self.category}".strip()

    @property
    def explanation(self) -> str:
        """The full plain-English explanation, reasons first."""
        lines: list[str] = []
        if self.reasons:
            lines.append("#### Why this category:")
            lines.extend(f"- {reason}" for reason in self.reasons)
        if self.observations:
            lines.append("")
            lines.append("#### What else stands out:")
            lines.extend(f"- {note}" for note in self.observations)
        return "\n".join(lines) if lines else "Not enough data to explain."


def _assert_no_advice_language(text: str) -> None:
    """Guard against accidentally giving investment advice.

    Args:
        text: A message about to be shown to the user.

    Raises:
        ValueError: If the message contains buy/sell style language.
    """
    lowered = text.lower()
    for word in _FORBIDDEN_WORDS:
        if word in lowered:
            raise ValueError(f"Advice-like language '{word}' must not appear in {text!r}")


def assess_suitability(
    ticker: str,
    prices: pd.Series,
    fast_window: int = config.SMA_FAST_WINDOW,
    slow_window: int = config.SMA_SLOW_WINDOW,
    rsi_window: int = config.RSI_WINDOW,
    warmup_prices: pd.Series | None = None,
) -> SuitabilityResult:
    """Assess how a stock has behaved and how risky it has been.

    The rules, in the order they are applied:

    1. **Not enough data** - fewer than 21 usable price rows.
    2. **High-Volatility Caution** - annualised volatility at or above 55%,
       or a maximum drawdown of -40% or worse.
    3. **Cautious / Steady** - annualised volatility at or below 25% *and* a
       maximum drawdown better than -25%.
    4. **Momentum / Watchful** - a total return of at least 25% with the price
       above its 50-day average.
    5. **Balanced / Typical** - everything else.

    Args:
        ticker: Symbol being assessed (used in the messages).
        prices: Adjusted close series for the selected window. Return,
            volatility and drawdown are measured over exactly this series.
        fast_window: Window for the short moving average.
        slow_window: Window for the long moving average.
        rsi_window: Window for RSI.
        warmup_prices: Optional longer history used **only** to calculate the
            moving averages and RSI, so that they are already populated on the
            first day of a short window.

    Returns:
        A :class:`SuitabilityResult` with the category, the triggered reasons
        and the underlying numbers.
    """
    clean = prices.dropna()
    if len(clean) < config.MIN_HISTORY_ROWS:
        result = SuitabilityResult(
            ticker=ticker,
            category="Not Enough Data",
            summary=config.SUITABILITY_CATEGORIES["Not Enough Data"]["summary"],
            reasons=[
                f"Only {len(clean)} usable price rows are available, but at least "
                f"{config.MIN_HISTORY_ROWS} are needed before these measures mean anything."
            ],
        )
        result.emoji = config.SUITABILITY_CATEGORIES["Not Enough Data"]["emoji"]
        result.colour = config.SUITABILITY_CATEGORIES["Not Enough Data"]["colour"]
        return result

    returns = daily_returns(clean)
    total_return = total_return_pct(clean)
    annual_return = annualised_return_pct(clean)
    volatility = annualised_volatility_pct(returns)
    drawdown = maximum_drawdown(clean)
    worst_day = worst_day_pct(returns)

    indicator_source = clean
    if warmup_prices is not None:
        extended = warmup_prices.dropna()
        # Only use the extra history if it really does extend the window.
        if len(extended) > len(clean) and extended.index[0] < clean.index[0]:
            indicator_source = extended

    sma_fast = simple_moving_average(indicator_source, fast_window)
    sma_slow = simple_moving_average(indicator_source, slow_window)
    latest_price = float(clean.iloc[-1])
    latest_fast = float(sma_fast.dropna().iloc[-1]) if not sma_fast.dropna().empty else float("nan")
    latest_slow = float(sma_slow.dropna().iloc[-1]) if not sma_slow.dropna().empty else float("nan")
    rsi_series = rsi(indicator_source, rsi_window)
    rsi_clean = rsi_series.dropna()
    latest_rsi = float(rsi_clean.iloc[-1]) if not rsi_clean.empty else float("nan")

    metrics = {
        "total_return_pct": total_return,
        "annualised_return_pct": annual_return,
        "volatility_pct": volatility,
        "max_drawdown_pct": drawdown.max_drawdown_pct,
        "worst_day_pct": worst_day,
        "rsi": latest_rsi,
        "sma_fast": latest_fast,
        "sma_slow": latest_slow,
        "price": latest_price,
        "rows": float(len(clean)),
    }

    reasons: list[str] = []
    category = "Balanced / Typical"

    # --- Rule 2: high volatility or a severe historical fall -------------- #
    high_vol = volatility >= config.VOL_HIGH_PCT
    severe_drawdown = drawdown.max_drawdown_pct <= config.DRAWDOWN_SEVERE_PCT
    if high_vol or severe_drawdown:
        category = "High-Volatility Caution"
        if high_vol:
            reasons.append(
                f"Annualised volatility is {volatility:.1f}%, which is at or above the "
                f"{config.VOL_HIGH_PCT:.0f}% threshold for this category. In plain terms, "
                f"this price moves around far more than a typical large-company share."
            )
        if severe_drawdown:
            reasons.append(
                f"The worst peak-to-trough fall in this period was "
                f"{drawdown.max_drawdown_pct:.1f}%, which is at or beyond the "
                f"{config.DRAWDOWN_SEVERE_PCT:.0f}% severe-decline threshold. "
                + drawdown.recovery_text
            )
    # --- Rule 3: calm and shallow ---------------------------------------- #
    else:
        calm = volatility <= config.VOL_LOW_PCT
        shallow = drawdown.max_drawdown_pct > config.DRAWDOWN_CAUTION_PCT
        if calm and shallow:
            category = "Cautious / Steady"
            reasons.append(
                f"Annualised volatility is {volatility:.1f}%, at or below the "
                f"{config.VOL_LOW_PCT:.0f}% threshold - the price has been relatively calm."
            )
            reasons.append(
                f"The worst decline in this period was {drawdown.max_drawdown_pct:.1f}%, "
                f"which stayed inside the {abs(config.DRAWDOWN_CAUTION_PCT):.0f}% "
                "caution threshold. " + drawdown.recovery_text
            )
        else:
            # --- Rule 4: strong momentum ---------------------------------- #
            strong_trend = total_return >= config.RETURN_MOMENTUM_PCT and latest_price > latest_slow
            if strong_trend:
                category = "Momentum / Watchful"
                reasons.append(
                    f"The price gained {total_return:+.1f}% over this window, at or above "
                    f"the {config.RETURN_MOMENTUM_PCT:.0f}% momentum threshold."
                )
                reasons.append(
                    f"The latest close ({latest_price:,.2f}) is above its "
                    f"{slow_window}-day average ({latest_slow:,.2f}), so the upward "
                    "trend is still intact by this measure."
                )
            else:
                reasons.append(
                    f"Annualised volatility of {volatility:.1f}% sits between the "
                    f"{config.VOL_LOW_PCT:.0f}% and {config.VOL_HIGH_PCT:.0f}% thresholds, "
                    f"and the total return of {total_return:+.1f}% does not reach the "
                    f"{config.RETURN_MOMENTUM_PCT:.0f}% momentum threshold - so this "
                    "behaves like a typical large-company share."
                )

    # --- Observations that are always worth reporting ---------------------- #
    observations: list[str] = []
    observations.append(f"Trend check: {describe_trend(latest_price, latest_fast, latest_slow)}.")

    if latest_rsi == latest_rsi:  # not NaN
        if latest_rsi >= config.RSI_OVERBOUGHT:
            observations.append(
                f"RSI is {latest_rsi:.0f}, above {config.RSI_OVERBOUGHT:.0f}. After a run "
                "like this, momentum can fade quickly - it is a reading to watch, not a "
                "signal on its own."
            )
        elif latest_rsi <= config.RSI_OVERSOLD:
            observations.append(
                f"RSI is {latest_rsi:.0f}, below {config.RSI_OVERSOLD:.0f}. Weak momentum "
                "of this kind can last a long time before prices recover."
            )
        else:
            observations.append(
                f"RSI is {latest_rsi:.0f}, comfortably inside the "
                f"{config.RSI_OVERSOLD:.0f}-{config.RSI_OVERBOUGHT:.0f} band, so momentum "
                "is neither stretched up nor washed out."
            )

    observations.append(
        f"Worst single day in this period: {worst_day:.1f}%. That is the scale of a "
        "bad-day move a beginner should expect."
    )

    if len(clean) < config.RELIABLE_HISTORY_ROWS:
        observations.append(
            f"Reliability note: this window holds only {len(clean)} trading days, so the "
            "volatility and drawdown figures are rough estimates rather than precise "
            "measurements. Longer windows give steadier numbers."
        )
    observations.append(
        f"Over the full window the price returned {total_return:+.1f}%, which annualises "
        f"to {annual_return:+.1f}% a year."
    )

    if abs(total_return) >= config.RETURN_MOMENTUM_PCT:
        observations.append(
            f"Note on reading this label: the category describes how bumpy the journey "
            f"was, not how far the price travelled. Over this window the price moved "
            f"{total_return:+.1f}%."
        )

    summary = config.SUITABILITY_CATEGORIES[category]["summary"]
    result = SuitabilityResult(
        ticker=ticker,
        category=category,
        summary=summary,
        reasons=reasons,
        observations=observations,
        metrics=metrics,
        emoji=config.SUITABILITY_CATEGORIES[category]["emoji"],
        colour=config.SUITABILITY_CATEGORIES[category]["colour"],
    )
    _assert_no_advice_language(result.explanation)
    return result


def assess_many(
    price_map: dict[str, pd.Series],
    warmup_map: dict[str, pd.Series] | None = None,
) -> dict[str, SuitabilityResult]:
    """Assess several tickers at once.

    Args:
        price_map: Mapping of ticker symbol to its window price series.
        warmup_map: Optional mapping of ticker symbol to a longer history used
            only to warm up the moving averages and RSI.

    Returns:
        A mapping of ticker symbol to :class:`SuitabilityResult`.
    """
    results: dict[str, SuitabilityResult] = {}
    for ticker, prices in price_map.items():
        warmup = (warmup_map or {}).get(ticker)
        results[ticker] = assess_suitability(ticker, prices, warmup_prices=warmup)
    return results


def category_colour(category: str) -> str:
    """Return the hex colour associated with a suitability category."""
    return config.SUITABILITY_CATEGORIES.get(category, {}).get("colour", config.COLOR_NEUTRAL)
