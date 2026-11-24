"""Short-term forecasting with a simple, explainable model.

The model is a **ridge regression** on a handful of technical features. It is
deliberately simple: it runs instantly, every input is readable, and it is
strong enough to make an honest point about how much (or how little) short-term
forecasting can achieve.

Evaluation is strictly chronological - train on the past, test on the future -
and the results are reported together with a naive baseline so a reader can see
whether the model adds anything at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from . import config


@dataclass
class ForecastResult:
    """Everything produced by one forecasting run.

    Attributes:
        ticker: Symbol that was forecast.
        model_name: Human-readable model description.
        test_metrics: Metric name -> value for the held-out test period.
        baseline_metrics: The same metrics for the naive "no change" forecast.
        train_size: Number of training rows.
        test_size: Number of test rows.
        future_index: Dates of the projected days.
        future_close: Central forecast of the closing price.
        lower_band: Lower edge of the prediction band.
        upper_band: Upper edge of the prediction band.
        last_actual_close: The last observed close (the anchor for the forecast).
        feature_names: Human-readable names of the model inputs.
        coefficients: Fitted coefficient per feature (standardised units).
        residual_std: Typical size of the model's past errors, used for the band.
        direction_accuracy: Share of test days where up/down was called correctly.
        bias: Average signed error on the test set.
        errors: Any problems encountered (for example "not enough history").
    """

    ticker: str
    model_name: str = "Ridge regression on lagged returns and technical features"
    test_metrics: dict[str, float] = field(default_factory=dict)
    baseline_metrics: dict[str, float] = field(default_factory=dict)
    train_size: int = 0
    test_size: int = 0
    future_index: pd.DatetimeIndex | None = None
    future_close: pd.Series | None = None
    lower_band: pd.Series | None = None
    upper_band: pd.Series | None = None
    last_actual_close: float = float("nan")
    feature_names: list[str] = field(default_factory=list)
    coefficients: pd.Series | None = None
    residual_std: float = float("nan")
    direction_accuracy: float = float("nan")
    bias: float = float("nan")
    errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """True when the model produced a usable forecast."""
        return self.future_close is not None and not self.future_close.empty

    @property
    def beats_baseline(self) -> bool:
        """True when the model has a lower RMSE than the naive forecast."""
        model_rmse = self.test_metrics.get("RMSE ($)")
        base_rmse = self.baseline_metrics.get("RMSE ($)")
        if model_rmse is None or base_rmse is None:
            return False
        if any(value != value for value in (model_rmse, base_rmse)):
            return False
        return float(model_rmse) < float(base_rmse)


def build_features(prices: pd.Series, n_lags: int = config.N_LAGS) -> pd.DataFrame:
    """Build the model's input features from a price series.

    Features (all derived from information available *before* the day we
    predict):

    * ``return_lag_1 ... return_lag_n`` - the previous few days' percentage
      returns, which capture short-term momentum and mean reversion.
    * ``sma_ratio`` - today's close divided by its 20-day average minus one,
      i.e. "how stretched is the price right now".
    * ``sma_spread`` - the 20-day average divided by the 50-day average minus
      one, i.e. "which way is the trend pointing".
    * ``rsi_14`` - the momentum oscillator, scaled to 0-1.
    * ``vol_10`` - recent 10-day annualised volatility, in percent.
    * ``volume_change`` - today's volume relative to its 20-day average.

    Args:
        prices: Adjusted close series indexed by date.
        n_lags: How many lagged returns to include.

    Returns:
        A DataFrame of features indexed by date. The first rows are NaN
        because the indicators are not yet available.
    """
    frame = pd.DataFrame(index=prices.index)
    returns = prices.pct_change()
    for lag in range(1, n_lags + 1):
        frame[f"return_lag_{lag}"] = returns.shift(lag)

    sma_fast = prices.rolling(20, min_periods=10).mean()
    sma_slow = prices.rolling(50, min_periods=25).mean()
    frame["sma_ratio"] = prices / sma_fast - 1.0
    frame["sma_spread"] = sma_fast / sma_slow - 1.0

    delta = prices.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta).clip(lower=0.0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = gain / loss.replace(0.0, np.nan)
    frame["rsi_14"] = (100.0 - 100.0 / (1.0 + rs)) / 100.0

    frame["vol_10"] = returns.rolling(10, min_periods=5).std(ddof=1) * np.sqrt(252) * 100.0
    return frame


def _future_dates(last_date: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    """Approximate the next ``horizon`` business days after ``last_date``."""
    return pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=horizon)


def forecast(
    prices: pd.Series,
    ticker: str = "",
    horizon: int = config.FORECAST_HORIZON_DAYS,
    train_fraction: float = config.TRAIN_FRACTION,
    alpha: float = config.RIDGE_ALPHA,
    n_lags: int = config.N_LAGS,
    confidence: float = config.CONFIDENCE_LEVEL,
) -> ForecastResult:
    """Fit the model on the past and project the next few days.

    The pipeline is:

    1. Build features and the target (tomorrow's percentage return).
    2. Drop rows with missing values.
    3. Split **chronologically** - oldest 80% train, newest 20% test.
    4. Standardise the features and fit a ridge regression.
    5. Evaluate on the test set with MAE, RMSE and direction accuracy, and
       compare against a naive "tomorrow equals today" baseline.
    6. Refit on all data and project ``horizon`` days forward, adding a
       prediction band based on the size of the model's past errors.

    Args:
        prices: Adjusted close series indexed by date.
        ticker: Symbol being forecast (used for labelling).
        horizon: Number of trading days to project.
        train_fraction: Share of history used for training.
        alpha: Ridge regularisation strength.
        n_lags: Number of lagged returns used as features.
        confidence: Confidence level of the prediction band.

    Returns:
        A :class:`ForecastResult`. If there is not enough history the result
        carries an error message and ``is_valid`` is ``False``.
    """
    result = ForecastResult(ticker=ticker)

    clean = prices.dropna()
    if len(clean) < config.MIN_ROWS_FOR_FORECAST:
        result.errors.append(
            f"Only {len(clean)} price rows are available; at least "
            f"{config.MIN_ROWS_FOR_FORECAST} are needed to train and test a model."
        )
        return result

    features = build_features(clean, n_lags)
    target = clean.pct_change().shift(-1)  # tomorrow's return, known today
    dataset = features.join(target.rename("target")).dropna()

    if len(dataset) < 60:
        result.errors.append(
            f"Only {len(dataset)} complete feature rows are available after "
            "indicator warm-up; more history is needed."
        )
        return result

    split = int(len(dataset) * train_fraction)
    if split < 30 or len(dataset) - split < 10:
        result.errors.append("The history is too short to split into training and test sets.")
        return result

    x_train = dataset.iloc[:split].drop(columns=["target"])
    y_train = dataset.iloc[:split]["target"]
    x_test = dataset.iloc[split:].drop(columns=["target"])
    y_test = dataset.iloc[split:]["target"]

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = Ridge(alpha=alpha)
    model.fit(x_train_scaled, y_train)

    predictions = model.predict(x_test_scaled)
    result.train_size = len(x_train)
    result.test_size = len(x_test)

    # ---- Evaluation on the unseen test period --------------------------- #
    test_prices = clean.reindex(x_test.index)
    actual_next_close = test_prices.shift(-1).dropna()
    predicted_close = test_prices * (1.0 + pd.Series(predictions, index=x_test.index))
    common = actual_next_close.index.intersection(predicted_close.index)
    errors = predicted_close.reindex(common) - actual_next_close.reindex(common)

    if len(errors) > 0:
        result.test_metrics = {
            "MAE ($)": float(np.abs(errors).mean()),
            "RMSE ($)": float(np.sqrt((errors**2).mean())),
        }
        naive_errors = test_prices.reindex(common) - actual_next_close.reindex(common)
        result.baseline_metrics = {
            "MAE ($)": float(np.abs(naive_errors).mean()),
            "RMSE ($)": float(np.sqrt((naive_errors**2).mean())),
        }
        predicted_returns = pd.Series(predictions, index=x_test.index).reindex(common)
        realised_returns = actual_next_close.reindex(common) / test_prices.reindex(common) - 1.0
        direction = np.sign(predicted_returns) == np.sign(realised_returns)
        result.direction_accuracy = float(direction.mean() * 100.0)
        result.bias = float((predicted_returns - realised_returns).mean() * 100.0)

    # ---- Refit on everything and project forward ------------------------ #
    scaler_full = StandardScaler()
    x_full_scaled = scaler_full.fit_transform(dataset.drop(columns=["target"]))
    model_full = Ridge(alpha=alpha)
    model_full.fit(x_full_scaled, dataset["target"])

    train_residuals = y_train - model.predict(x_train_scaled)
    result.residual_std = float(np.std(train_residuals, ddof=1)) * 100.0  # in percent

    future_dates = _future_dates(clean.index[-1], horizon)
    current_price = float(clean.iloc[-1])
    forecast_returns: list[float] = []
    simulated = clean.copy()

    for offset in range(horizon):
        # Recompute the feature row from the simulated history so that every
        # projected day only uses information that would have existed by then.
        new_features = _features_for_single_point(simulated, n_lags)
        if new_features is None:
            break
        # Present the row as a DataFrame with the fitted feature names so the
        # scaler can validate it exactly as it did during training.
        feature_row = pd.DataFrame([new_features], columns=list(dataset.drop(columns=["target"]).columns))
        scaled = scaler_full.transform(feature_row)
        predicted_return = float(model_full.predict(scaled)[0])
        predicted_return = float(np.clip(predicted_return, -0.15, 0.15))  # sanity guard
        forecast_returns.append(predicted_return)
        simulated = pd.concat(
            [simulated, pd.Series([simulated.iloc[-1] * (1.0 + predicted_return)], index=[future_dates[offset]])]
        )

    if not forecast_returns:
        result.errors.append("The forecast horizon could not be extended - not enough history.")
        return result

    forecast_close: list[float] = []
    price = current_price
    for predicted_return in forecast_returns:
        price *= 1.0 + predicted_return
        forecast_close.append(price)

    # ---- Prediction band ------------------------------------------------- #
    from scipy.stats import norm  # local import: only needed when forecasting

    z_score = float(norm.ppf(0.5 + confidence / 2.0))
    band_width = z_score * result.residual_std / 100.0  # in return terms
    lower = [close * (1.0 - band_width) for close in forecast_close]
    upper = [close * (1.0 + band_width) for close in forecast_close]

    result.future_index = future_dates[: len(forecast_close)]
    result.future_close = pd.Series(forecast_close, index=result.future_index, name="Forecast")
    result.lower_band = pd.Series(lower, index=result.future_index, name="Lower")
    result.upper_band = pd.Series(upper, index=result.future_index, name="Upper")
    result.last_actual_close = current_price
    result.feature_names = list(x_train.columns)
    result.coefficients = pd.Series(model_full.coef_, index=x_train.columns).sort_values(
        key=abs, ascending=False
    )
    return result


def _features_for_single_point(history: pd.Series, n_lags: int) -> np.ndarray | None:
    """Compute the feature row for the next (unseen) day.

    Args:
        history: Price history including the most recent day.
        n_lags: Number of lagged returns used as features.

    Returns:
        A 1-D array of feature values, or ``None`` if they cannot be computed.
    """
    prices = history.dropna()
    if len(prices) < 60:
        return None

    returns = prices.pct_change()
    values: list[float] = []
    for lag in range(1, n_lags + 1):
        lagged = returns.shift(lag).iloc[-1]
        if lagged != lagged:
            return None
        values.append(float(lagged))

    sma_fast = prices.rolling(20, min_periods=10).mean().iloc[-1]
    sma_slow = prices.rolling(50, min_periods=25).mean().iloc[-1]
    price = prices.iloc[-1]
    if sma_fast != sma_fast or sma_slow != sma_slow or sma_fast == 0:
        return None
    values.append(float(price / sma_fast - 1.0))
    values.append(float(sma_fast / sma_slow - 1.0))

    delta = prices.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().iloc[-1]
    loss = (-delta).clip(lower=0.0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().iloc[-1]
    if loss == 0:
        values.append(1.0)
    else:
        rs = gain / loss
        values.append(float((100.0 - 100.0 / (1.0 + rs)) / 100.0))

    vol = returns.rolling(10, min_periods=5).std(ddof=1).iloc[-1]
    if vol != vol:
        return None
    values.append(float(vol * np.sqrt(252) * 100.0))

    return np.asarray(values, dtype="float64")


LIMITATIONS: tuple[str, ...] = (
    "**Market noise.** Daily price changes are mostly the result of thousands of "
    "small, unrelated decisions. No historical pattern can capture a surprise "
    "news headline, so a large share of day-to-day movement is simply "
    "unpredictable.",
    "**News shocks.** Earnings, interest-rate decisions, regulation and geopolitics "
    "can move a price far beyond anything in the training history.",
    "**Non-stationarity.** The market's behaviour changes over time. A relationship "
    "that held in a past regime may not hold now, which is why we always test on "
    "the most recent 20% of history rather than on random data.",
    "**Small samples.** A one-year window gives only a few hundred trading days. "
    "That is enough to show how a method behaves, not enough to prove it will keep "
    "working.",
    "**Forecasts compound.** A 1% daily prediction error is roughly 5% over a week "
    "and 30% over a year. Short horizons are the only ones where such a model is "
    "defensible at all.",
    "**No certainty.** The shaded band is a statistical guide built from the model's "
    "own past mistakes. Real prices can and do fall outside it.",
)


def forecast_explanation(result: ForecastResult) -> list[str]:
    """Turn a forecast result into beginner-friendly sentences.

    Args:
        result: Output of :func:`forecast`.

    Returns:
        A list of plain-English sentences for the dashboard.
    """
    if not result.is_valid:
        return result.errors or ["A forecast could not be produced for this selection."]

    lines: list[str] = []
    if result.future_close is not None and result.future_index is not None:
        first_day = result.future_index[0].strftime("%d %b %Y")
        final_close = float(result.future_close.iloc[-1])
        change = (final_close / result.last_actual_close - 1.0) * 100.0
        direction = "higher" if change >= 0 else "lower"
        lines.append(
            f"Starting from the last close of {result.last_actual_close:,.2f}, the model "
            f"projects a close of about {final_close:,.2f} by "
            f"{result.future_index[-1].strftime('%d %b %Y')} - roughly {abs(change):.1f}% "
            f"{direction} over the {len(result.future_close)} trading days from {first_day}."
        )

    mae = result.test_metrics.get("MAE ($)")
    rmse = result.test_metrics.get("RMSE ($)")
    base_rmse = result.baseline_metrics.get("RMSE ($)")
    if mae is not None and rmse is not None:
        lines.append(
            f"On the {result.test_size} unseen test days the model's average mistake was "
            f"${mae:,.2f} per share (MAE) and its typical squared mistake corresponded to "
            f"${rmse:,.2f} (RMSE)."
        )
    if base_rmse is not None and rmse is not None:
        verdict = (
            "slightly better than"
            if result.beats_baseline
            else "no better than"
        )
        lines.append(
            f"That is {verdict} the naive 'tomorrow equals today' baseline, which scored "
            f"an RMSE of ${base_rmse:,.2f} on exactly the same days."
        )
    if result.direction_accuracy == result.direction_accuracy:
        lines.append(
            f"The model called the direction of the move correctly on "
            f"{result.direction_accuracy:.0f}% of test days (a coin toss would be 50%)."
        )
    if result.residual_std == result.residual_std:
        lines.append(
            f"Typical daily prediction error was about {result.residual_std:.2f}%, which is "
            "what sets the width of the shaded uncertainty band."
        )
    return lines
