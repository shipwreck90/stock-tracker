# StockLens 📈

**A beginner-friendly stock market tracker, performance comparator, risk/suitability analyzer and short-term forecaster — built for a university assignment.**

StockLens is a Streamlit web application that lets someone with **no trading experience** explore how a handful of stocks have performed, how risky each one has been, and what a simple statistical model says might happen next — with every technical term explained in plain English.

---

## ⚠️ Educational disclaimer

> **This tool is for academic and educational purposes only. It is not financial, investment, tax or legal advice, and nothing in it is a recommendation to buy, sell or hold any security.**
>
> * Forecasts are **probabilistic estimates, not guarantees**. A model that was accurate yesterday can be completely wrong tomorrow, because markets are driven by news, sentiment and human behaviour that no historical pattern contains.
> * Past performance does not predict future results.
> * Data comes from a third party and may be delayed, incomplete or incorrect. Always verify figures against a trusted source.
> * You are responsible for any decisions you make.

The suitability engine is **descriptive, never prescriptive**: it labels how a stock has *behaved* and shows every number behind that label. There is no "buy" or "sell" output anywhere in the codebase, and an automated test in `run_demo.py` enforces this.

---

## Quick start

```bash
# 1. Get the code and move into the project folder
cd stock-tracker

# 2. Create a virtual environment (recommended)
python -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate

# 3. Install the dependencies
pip install -r requirements.txt

# 4. Launch the dashboard
streamlit run app.py
```

Your browser opens automatically at `http://localhost:8501`.

**No API key is needed.** Prices come from Yahoo Finance's public chart service.

### Prefer the terminal? Run the headless demo

```bash
python run_demo.py
python run_demo.py --tickers AAPL,MSFT,KO --benchmark ^GSPC --timeframe "6 Months"
```

This prints a full text report (leaderboard, suitability reasoning, forecast with error metrics) and runs five automated self-checks. It is handy for a viva demonstration and for capturing output in your report.

---

## Project structure

```
stock-tracker/
├── app.py                  # Streamlit dashboard (the user interface)
├── run_demo.py             # Headless demo + automated self-checks
├── requirements.txt        # Pinned-ish dependency list
├── README.md               # This file
│
├── stocklens/              # The analysis package
│   ├── config.py           # Every default, threshold and file path
│   ├── glossary.py         # "Jargon Translator" - plain-English definitions
│   ├── data_loader.py      # Download, validate, cache, CSV fallback
│   ├── indicators.py       # SMA, RSI, volatility, drawdown, returns
│   ├── performance.py      # Leaderboard, rebasing, benchmark-relative returns
│   ├── suitability.py      # Rule-based risk/suitability assessment
│   ├── forecaster.py       # Ridge-regression forecast + evaluation
│   ├── viz.py              # All Plotly chart builders
│   └── sample_data.py      # Synthetic offline data generator
│
└── data/                   # Created automatically
    ├── cache/              # Downloaded price CSVs (reused for 6 hours)
    ├── user/               # ← Put your own CSV files here
    └── sample/             # Bundled synthetic sample data (clearly labelled)
```

### Data flow

```
                ┌──────────────┐
   Yahoo  ───▶ │ data_loader  │ ◀─── CSV in data/user/
   Finance      │  (validate + │ ───▶ synthetic sample (offline only)
                │   cache)     │
                └──────┬───────┘
                       │  adjusted-close frame
          ┌────────────┼─────────────┐
          ▼            ▼             ▼
   ┌────────────┐ ┌──────────┐ ┌────────────┐
   │indicators  │ │performance│ │ forecaster │
   │ SMA, RSI,  │ │ leader-  │ │ ridge model│
   │ vol, DD    │ │ board    │ │ + MAE/RMSE │
   └─────┬──────┘ └────┬─────┘ └─────┬──────┘
         │             │              │
         ▼             ▼              ▼
   ┌───────────┐   ┌────────────────────────┐
   │suitability│──▶│  viz.py → app.py (UI)  │
   └───────────┘   └────────────────────────┘
```

---

## The five tabs

| Tab | The question it answers |
|---|---|
| 🏠 **Overview** | What am I even looking at? Summary cards, plain-English summary, rebased performance chart, downloadable table. |
| 📈 **Performance** | *Who is performing better or worse?* Leaderboard, returns bar chart, benchmark-relative chart, column-by-column definitions. |
| 🛡️ **Risk & Suitability** | *Which looks stronger vs riskier for a beginner?* Rule-based profile with every reason shown, price/SMA chart, RSI chart, drawdown chart, and a table of all selected tickers. |
| 🔮 **Forecast** | *What might happen next?* 5-day projection with an uncertainty band, MAE / RMSE / direction accuracy, baseline comparison, feature weights, and an explicit limitations list. |
| 📖 **Glossary & Methodology** | *What does all this jargon mean?* Every term explained (plain English, why it matters, how it is computed), plus the full methodology and reproducibility notes. |

---

## How to interpret the outputs

### 1. Performance comparison

* **Total return (%)** — the simple percentage change across the window. This is the headline "who won" number.
* **Annualised return (%/yr)** — the yearly growth rate implied by the window, so a 1-month and a 3-year selection can be compared fairly.
* **Return vs benchmark (pp)** — percentage points ahead of (or behind) the benchmark over the *same days*. A stock can be up while the market is down; this column is what separates the company's result from the market's mood.
* **Rebased to 100** — every series restarts at 100, so a $20 stock and a $500 stock can share one chart. The line that ends highest is the best performer.

### 2. Risk and suitability

Each ticker receives one of five **rule-based** categories, chosen by fixed thresholds that are printed in the UI:

| Category | Trigger |
|---|---|
| 🛡️ **Cautious / Steady** | Annualised volatility ≤ 25% **and** worst drawdown better than −25% |
| ⚖️ **Balanced / Typical** | Everything else |
| 🚀 **Momentum / Watchful** | Total return ≥ 25% **and** price above its 50-day average |
| ⚠️ **High-Volatility Caution** | Annualised volatility ≥ 55% **or** a drawdown of −40% or worse |
| ❓ **Not Enough Data** | Fewer than 21 usable price rows |

Read the **"Why"** lines, not just the badge: each one names the threshold and the actual number, so you can check the reasoning or disagree with it. The label describes *how bumpy the journey was*, not how far the price travelled — a stock can be "Cautious / Steady" and still have risen sharply.

Key measures used:

* **Annualised volatility** — the standard deviation of daily returns, scaled up to a year. Think of it as the typical size of a daily surprise, multiplied up.
* **Maximum drawdown** — the worst peak-to-trough fall in the window. This is the "how bad did it get, and for how long was I underwater?" number, and it matters far more to a beginner than the average day.
* **RSI (14-day)** — a 0–100 oscillator comparing the size of recent up-days with recent down-days. Above 70 is conventionally read as "stretched", below 30 as "beaten down". Both are *guides*, never signals.
* **20-day / 50-day SMA** — rolling averages that smooth out daily noise so the trend becomes visible.

### 3. The forecast

The model is a **ridge regression** — a simple, explainable linear model with a small penalty that stops it latching onto noise. Its inputs are the last 5 daily returns, the price relative to its 20- and 50-day averages, RSI, recent volatility, and volume change.

* **Chronological split.** The oldest 80% of the history trains the model; the newest 20% is held back and never seen during training. Nothing is shuffled — shuffling would leak information from the future into the past and make the score look unrealistically good.
* **MAE** — average size of the prediction error, in dollars per share. The most intuitive accuracy number.
* **RMSE** — like MAE, but it punishes occasional huge mistakes much harder. Always ≥ MAE.
* **Direction accuracy** — how often the model correctly said "up" or "down". 50% is a coin toss.
* **Naive baseline** — "tomorrow's price equals today's". A model is only interesting if it beats this trivial guess on unseen data. Sometimes it does not, and *that is a legitimate and valuable finding* to report.
* **Prediction band** — the shaded range around the forecast, built from the model's own historical error size at a 95% confidence level. It shows scale, not certainty.

**Model limitations (also listed in the app):** market noise, news shocks, non-stationarity, small samples, compounding error over longer horizons, and the simple fact that real prices escape the band regularly.

---

## Methodology notes

| Step | Detail |
|---|---|
| **Data** | Daily OHLCV from Yahoo Finance's public chart service; `yfinance` as an alternative loader; local CSV in `data/user/` as a fallback; deterministic synthetic data only for the five known sample symbols and always clearly labelled. |
| **Validation** | Duplicate dates removed, missing closes dropped, non-positive prices discarded, adjusted close substituted when unavailable, and one-day moves above ±80% flagged as probable data errors. |
| **Caching** | Each symbol is cached as CSV in `data/cache/` for 6 hours. A cache that does not reach far enough back for the selected timeframe is rejected and re-downloaded. |
| **Window** | Exactly the selected number of trading days, dated from the benchmark series so every ticker is compared over identical days. Extra warm-up history is fetched so the 50-day average is populated on the first visible day. |
| **Returns** | Based on adjusted closes where available, so dividends and splits are handled fairly. |
| **Annualisation** | 252 trading days per year. |
| **Model** | Ridge regression (`alpha=1.0`), features standardised with `StandardScaler`, deterministic — the same inputs always give the same output. |
| **Reproducibility** | Fixed random seed for the synthetic data generator; no shuffling anywhere; every threshold lives in `config.py`. |

---

## Using your own data

Drop a CSV into `data/user/` named after the ticker (for example `data/user/BA.csv`):

```csv
Date,Open,High,Low,Close,Volume
2024-01-02,154.32,155.10,152.80,153.18,8123400
2024-01-03,153.20,154.90,152.44,154.11,6932100
```

A `Date` column and a `Close` column are the minimum; `Adj Close` is used when present. Files in `data/user/` take priority over the live download, which makes the app work offline and lets you demo with a fixed dataset.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` inside your activated virtual environment. |
| "No price history could be found" | The symbol is misspelled, delisted, or not covered by the provider. Try a different symbol or add a CSV to `data/user/`. |
| "SYNTHETIC sample data" banner | The live download failed. Check your connection, then press **🔄 Refresh data now**. |
| Numbers differ between runs | Market data updates daily. Use the same day for a fair comparison, or download the tables as CSV. |
| Port 8501 already in use | `streamlit run app.py --server.port 8502` |
| Chart is empty for a short timeframe | The 50-day average needs history; the app fetches warm-up data automatically, but a ticker with fewer than 60 days of history will have gaps. |

---

## Ideas for extending the project

These are natural next steps, each of which would strengthen a report:

1. **Backtest a simple strategy** (e.g. "buy when the 20-day average crosses above the 50-day") and compare it with simply holding the stock — including transaction costs.
2. **Add a second benchmark** or a sector comparison.
3. **Compare models** — naive, linear regression, ridge with different `alpha` values, and a random walk — using the same chronological split.
4. **Bootstrap the confidence band** instead of relying on the normal approximation.
5. **Add a walk-forward (rolling) evaluation** so the model is retested at several points in time rather than only once.
6. **Add unit tests** with `pytest` over the indicator functions using hand-calculated values.

---

## Academic honesty note

If you submit this for an assignment, cite it and be explicit about what you built yourself. The data is real (from Yahoo Finance, subject to their terms), the synthetic fallback is clearly labelled, and every number in the app can be traced back to a documented formula in this README and in the docstrings of the module that computes it. Marker-friendly summary: the value of this project is not the forecasts — it is the transparent, jargon-free explanation of *how* the numbers were produced and *why* they should not be trusted blindly.

---

## Licence and data use

The code is provided for educational use. Market data belongs to Yahoo Finance and is subject to Yahoo's terms of service; the app caches it locally for personal, non-commercial use only.
