"""The "Jargon Translator": plain-English explanations for every metric.

The whole point of this project is that a beginner can understand what the
numbers mean. Each entry below is keyed by a stable metric id that the rest of
the code base uses, so the explanation shown next to a number can never drift
away from the calculation that produced it.

Each entry contains four parts:

``term``
    The name of the metric as it appears on screen.
``plain``
    One or two sentences with no jargon at all.
``why``
    Why a beginner should care about it.
``how``
    Exactly how this app computes it (formula in words, no notation required).

:func:`glossary_dataframe` renders the whole dictionary as a table for the
"Glossary & Methodology" tab of the dashboard.
"""

from __future__ import annotations

import pandas as pd

#: Metric id -> explanation. Keys are used by the UI to attach tooltips.
GLOSSARY: dict[str, dict[str, str]] = {
    # ---------------- Data basics ---------------- #
    "ticker": {
        "term": "Ticker",
        "plain": "The short code a company or fund trades under, like AAPL for Apple.",
        "why": "It is the name you type into any market app to find the same asset.",
        "how": "We look up each ticker on Yahoo Finance and ask for its daily prices.",
    },
    "benchmark": {
        "term": "Benchmark",
        "plain": (
            "A stand-in for 'the market as a whole'. Here it is the S&P 500, an index "
            "of 500 large US companies."
        ),
        "why": (
            "A stock can rise while the whole market falls. The benchmark is how you "
            "tell the difference between the company's result and the market's mood."
        ),
        "how": (
            "We compare each ticker's return with the benchmark's return over exactly "
            "the same days, so the comparison is fair."
        ),
    },
    "price": {
        "term": "Closing price",
        "plain": "The price the stock finished at when trading closed for the day.",
        "why": "It is the standard, official 'value' of the stock for each day.",
        "how": "Taken from the daily close of each trading session.",
    },
    "adjusted_close": {
        "term": "Adjusted close",
        "plain": (
            "The closing price corrected for things that happen to shareholders but "
            "not to the share count, such as dividends and stock splits."
        ),
        "why": (
            "It is the fairer number for comparing returns over time, because a split "
            "does not actually make you richer."
        ),
        "how": "We use it whenever one is available and fall back to the plain close.",
    },
    "daily_return": {
        "term": "Daily return",
        "plain": "The percentage change in price from one day to the next.",
        "why": "Returns are how you measure success; price alone is harder to compare.",
        "how": "Today's price divided by yesterday's price, minus one.",
    },
    # ---------------- Performance ---------------- #
    "total_return": {
        "term": "Total return (%)",
        "plain": (
            "How much the price gained or lost across the whole period you picked, "
            "as a percentage of where it started."
        ),
        "why": "It is the single simplest answer to 'did this go up or down?'.",
        "how": "Final price divided by starting price, minus one, times 100.",
    },
    "annualised_return": {
        "term": "Annualised return (% per year)",
        "plain": (
            "The yearly speed of growth, worked out so that short periods and long "
            "periods can be compared fairly."
        ),
        "why": (
            "A 20% gain over a month is very different from 20% over five years. "
            "This number puts both on the same scale."
        ),
        "how": (
            "We take the total return, convert it to a daily compound rate, then "
            "multiply by the 252 trading days in a year."
        ),
    },
    "relative_return": {
        "term": "Return vs benchmark (percentage points)",
        "plain": (
            "How many percentage points the ticker earned more (or less) than the "
            "benchmark over the same days."
        ),
        "why": "This is the 'who beat the market?' number, stripped of market-wide noise.",
        "how": "The ticker's total return minus the benchmark's total return.",
    },
    "rebased": {
        "term": "Rebased to 100",
        "plain": (
            "Every line starts at the same point - 100 - so you can compare paths "
            "regardless of their actual price."
        ),
        "why": (
            "A 20-dollar and a 500-dollar stock look impossible to compare on a chart "
            "until you put them on the same starting footing."
        ),
        "how": (
            "Each series is divided by its first value in the window and multiplied "
            "by 100."
        ),
    },
    "leaderboard": {
        "term": "Leaderboard",
        "plain": "The tickers sorted from best to worst performing over your window.",
        "why": "It gives you the answer to 'who is performing better or worse?' at a glance.",
        "how": "Rows are ranked by total return, highest first.",
    },
    # ---------------- Trend and momentum ---------------- #
    "sma20": {
        "term": "20-day moving average (SMA)",
        "plain": (
            "The average price of the last 20 trading days. It moves up when recent "
            "prices are higher, and down when they are lower."
        ),
        "why": (
            "It smooths out daily noise so you can see the general direction of travel."
        ),
        "how": "Each day we average the last 20 closing prices and plot the result.",
    },
    "sma50": {
        "term": "50-day moving average (SMA)",
        "plain": "The same idea as the 20-day average, but over a longer two-month window.",
        "why": (
            "Together with the 20-day average it shows whether the short-term and "
            "medium-term trends agree."
        ),
        "how": "Each day we average the last 50 closing prices and plot the result.",
    },
    "trend_signal": {
        "term": "Trend signal",
        "plain": (
            "A short sentence describing where the price sits relative to its own "
            "recent averages."
        ),
        "why": "It converts a chart shape into a phrase a beginner can act on.",
        "how": (
            "We compare the latest close with the 20-day and 50-day averages and "
            "describe the combination."
        ),
    },
    "rsi": {
        "term": "14-day RSI (Relative Strength Index)",
        "plain": (
            "A number from 0 to 100 that answers: 'how much of the time has this "
            "stock been going up versus going down recently?'"
        ),
        "why": (
            "It is a quick way to see when a stock may have been pushed very hard in "
            "one direction."
        ),
        "how": (
            "We compare the average size of recent gains with the average size of "
            "recent losses, using a smoothing method (Wilder's) so the number does "
            "not jump around."
        ),
    },
    "rsi_bands": {
        "term": "RSI over 70 / under 30",
        "plain": (
            "Above 70 is commonly read as 'stretched' after a strong run; below 30 "
            "is commonly read as 'beaten down'."
        ),
        "why": "It hints at where a move might be running out of steam.",
        "how": "We use the traditional 70 and 30 cut-offs, and always say they are a "
               "guide rather than a signal.",
    },
    # ---------------- Risk ---------------- #
    "volatility": {
        "term": "Annualised volatility (%)",
        "plain": (
            "How bumpy the price has been, scaled up to a full year. Think of it as "
            "the typical size of a day's surprise, multiplied up."
        ),
        "why": (
            "Two stocks can return the same amount with very different journeys. "
            "Volatility measures the roughness of the journey."
        ),
        "how": (
            "We take the standard deviation of daily returns and multiply by the "
            "square root of 252 trading days."
        ),
    },
    "max_drawdown": {
        "term": "Maximum drawdown (%)",
        "plain": (
            "The worst peak-to-trough fall the price experienced during the period. "
            "'If you bought on the worst day, this is the biggest drop you could have "
            "seen before it recovered.'"
        ),
        "why": (
            "It is the honest measure of how bad things got, and it is what a "
            "beginner should care about more than the average day."
        ),
        "how": (
            "We track the highest price reached so far and measure the largest fall "
            "from that peak to a later low."
        ),
    },
    "risk_reward_ratio": {
        "term": "Return per unit of risk",
        "plain": (
            "How much return you earned for each unit of volatility you had to "
            "tolerate."
        ),
        "why": "It lets you compare two investments that paid differently but felt different.",
        "how": "Annualised return divided by annualised volatility.",
    },
    "worst_day": {
        "term": "Worst single day (%)",
        "plain": "The largest one-day fall in the period.",
        "why": "It shows what a bad day has actually looked like for this stock.",
        "how": "The smallest daily return observed in the window.",
    },
    # ---------------- Suitability rules ---------------- #
    "suitability": {
        "term": "Beginner suitability profile",
        "plain": (
            "A rule-based label describing how this stock has behaved and how risky "
            "it has been, with every reason shown."
        ),
        "why": (
            "It gives you a starting point for research - not a recommendation to "
            "buy, sell or hold anything."
        ),
        "how": (
            "Fixed rules look at volatility, drawdown, trend and momentum. Every "
            "triggered rule is listed with its numbers, so you can disagree with it."
        ),
    },
    "rule_based": {
        "term": "Rule-based assessment",
        "plain": (
            "The label comes from simple published thresholds, not from a black-box "
            "prediction."
        ),
        "why": "You can check the reasoning yourself, which is the whole point.",
        "how": "Each rule is a simple comparison such as 'volatility above 55%'.",
    },
    # ---------------- Forecasting ---------------- #
    "forecast": {
        "term": "Short-term forecast",
        "plain": (
            "The model's best guess at where the price may go over the next few "
            "trading days, shown as a central estimate with a shaded range."
        ),
        "why": (
            "It demonstrates how forecasting works and, just as importantly, how "
            "wrong such forecasts can be."
        ),
        "how": (
            "A regression model combines the last few days' returns with a few "
            "technical indicators to predict the next day's return, which is then "
            "compounded forward."
        ),
    },
    "ridge": {
        "term": "Ridge regression",
        "plain": (
            "A simple, well-behaved version of linear regression that adds a small "
            "penalty for using complicated coefficients. The penalty keeps the model "
            "from latching onto noise."
        ),
        "why": (
            "It is easy to explain to a marker, runs in milliseconds, and is a "
            "standard, respectable baseline for this kind of task."
        ),
        "how": (
            "We fit a straight-line relationship between the features and the next "
            "day's return, shrinking the coefficients slightly towards zero."
        ),
    },
    "features": {
        "term": "Features (model inputs)",
        "plain": (
            "The handful of pieces of information the model is allowed to look at: "
            "the last five daily returns, today's momentum readings and recent "
            "volatility."
        ),
        "why": "It shows that a forecast is only as good as its inputs.",
        "how": (
            "We build these numbers from the price history, shift them forward one "
            "day, and line them up with the day we want to predict."
        ),
    },
    "train_test_split": {
        "term": "Chronological train/test split",
        "plain": (
            "We teach the model on the older part of the history and then test it on "
            "the newer part, exactly as if it had never seen those days."
        ),
        "why": (
            "Mixing the order (a common mistake) would leak information from the "
            "future into the past and make the score look unrealistically good."
        ),
        "how": (
            "The oldest 80% of the days are training data; the newest 20% are test "
            "data. Nothing is shuffled."
        ),
    },
    "mae": {
        "term": "MAE (Mean Absolute Error)",
        "plain": (
            "On average, how far off the model's predicted price was, in dollars "
            "per share. Bigger number = less accurate."
        ),
        "why": "It is the easiest accuracy number for a beginner to interpret.",
        "how": "We average the size of every prediction error on the test period, ignoring "
               "whether it was too high or too low.",
    },
    "rmse": {
        "term": "RMSE (Root Mean Squared Error)",
        "plain": (
            "Like MAE, but it punishes the occasional huge mistake much harder. It "
            "is always at least as large as MAE."
        ),
        "why": (
            "It tells you whether the model is reliably okay or occasionally wildly "
            "wrong."
        ),
        "how": "We average the squared errors and then take the square root, to bring "
               "the units back to dollars.",
    },
    "baseline": {
        "term": "Baseline (naive forecast)",
        "plain": (
            "The simplest possible guess: 'tomorrow's price will be the same as "
            "today's'."
        ),
        "why": (
            "A model is only worth using if it beats this trivial guess on unseen "
            "data."
        ),
        "how": "We evaluate the naive forecast on exactly the same test days and "
               "compare the error numbers.",
    },
    "prediction_interval": {
        "term": "Prediction band (uncertainty range)",
        "plain": (
            "The shaded area around the forecast line showing the range the real "
            "price could plausibly fall inside."
        ),
        "why": (
            "A forecast without a range is misleading. This shows how uncertain the "
            "model really is."
        ),
        "how": (
            "We use the size of the model's typical past mistakes and a standard "
            "statistical rule to draw the band. It is a guide, not a promise."
        ),
    },
    "direction_accuracy": {
        "term": "Direction accuracy (%)",
        "plain": (
            "How often the model correctly said 'up' or 'down' compared with what "
            "actually happened."
        ),
        "why": "It answers 'is the model even useful?' in one number.",
        "how": "We compare the sign of each predicted return with the sign of the "
               "realised return.",
    },
    "model_limitations": {
        "term": "Model limitations",
        "plain": (
            "The honest list of reasons this forecast can be wrong: market noise, "
            "unexpected news, and the fact that markets change their habits over time."
        ),
        "why": (
            "A forecast presented without limits looks far more reliable than it is. "
            "Understanding the limits is the academic point."
        ),
        "how": "We measure error on unseen data and report it alongside the forecasts "
               "rather than hiding it.",
    },
    # ---------------- Methodology ---------------- #
    "data_source": {
        "term": "Data source",
        "plain": (
            "Daily prices come from Yahoo Finance's public chart service. If the "
            "network is unavailable we fall back to bundled sample files or your own "
            "CSV, and we tell you when that happens."
        ),
        "why": "You should always know where numbers came from.",
        "how": "We validate every download, cache it locally, and record which route "
               "the data actually took.",
    },
    "cache": {
        "term": "Local cache",
        "plain": (
            "Downloaded prices are saved on your computer for a few hours so the app "
            "starts instantly and still works if the network goes down."
        ),
        "why": "Faster demos, fewer repeated downloads, offline resilience.",
        "how": "We store each symbol's history as a CSV file with a timestamp.",
    },
    "non_stationarity": {
        "term": "Non-stationarity",
        "plain": (
            "The idea that market behaviour changes over time, so a relationship "
            "that worked years ago may not work today."
        ),
        "why": (
            "It is the single biggest reason backtests look wonderful and live "
            "trading does not."
        ),
        "how": "We keep the test set strictly in the future and report real error.",
    },
    "noise": {
        "term": "Market noise",
        "plain": (
            "The daily ups and downs that come from thousands of small decisions "
            "rather than from anything about the company."
        ),
        "why": "Noise is why most short-term forecasts are only slightly better than guesses.",
        "how": "Our model deliberately uses few inputs and penalises complexity to "
               "avoid fitting noise.",
    },
    "bias": {
        "term": "Prediction bias",
        "plain": (
            "If the model consistently guesses a little too high or a little too low. "
            "Small bias is normal; large bias is a warning sign."
        ),
        "why": "It tells you whether the errors are balanced or lopsided.",
        "how": "We subtract the average error from the test predictions.",
    },
    "no_advice": {
        "term": "No buy or sell calls",
        "plain": (
            "This tool never tells you to buy, sell or hold. It describes what "
            "happened and what a simple model predicts, then lets you decide."
        ),
        "why": "It keeps a learning project honest and avoids giving financial advice.",
        "how": "The rule engine produces descriptive categories only - there is no "
               "buy/sell output anywhere in the code.",
    },
}


def glossary_dataframe() -> pd.DataFrame:
    """Return the whole glossary as a tidy table for display.

    Returns:
        A DataFrame with the columns ``id``, ``term``, ``plain``, ``why`` and
        ``how``, sorted alphabetically by term.
    """
    rows = [
        {
            "id": key,
            "term": entry["term"],
            "plain": entry["plain"],
            "why": entry["why"],
            "how": entry["how"],
        }
        for key, entry in GLOSSARY.items()
    ]
    return pd.DataFrame(rows).sort_values("term").reset_index(drop=True)


def explain(metric_id: str) -> str:
    """Return a one-line plain-English explanation for a metric id.

    Args:
        metric_id: Key used in :data:`GLOSSARY`, for example ``"rsi"``.

    Returns:
        The plain-English sentence, or a short fallback if the id is unknown.
    """
    entry = GLOSSARY.get(metric_id)
    if entry is None:
        return "No glossary entry for this metric."
    return entry["plain"]


def get_entry(metric_id: str) -> dict[str, str]:
    """Return the full glossary record for a metric id (empty dict if unknown)."""
    return GLOSSARY.get(metric_id, {})
