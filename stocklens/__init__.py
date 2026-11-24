"""StockLens - a beginner-friendly stock market tracker and forecaster.

Public entry points:

* :func:`stocklens.data_loader.load_market_data` - fetch and validate prices.
* :func:`stocklens.indicators.indicator_panel` - technical indicators.
* :func:`stocklens.performance.build_performance_table` - leaderboard.
* :func:`stocklens.suitability.assess_suitability` - risk profile rules.
* :func:`stocklens.forecaster.forecast` - short-term model.
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = [
    "config",
    "glossary",
    "data_loader",
    "indicators",
    "performance",
    "suitability",
    "forecaster",
    "viz",
    "sample_data",
]
