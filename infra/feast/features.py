"""
Feast Feature Definitions for QuantCLI

This module defines all features in the feature store.
Features are versioned and registered with Feast for consistent
offline (training) and online (serving) access.
"""

from datetime import timedelta

from feast import Entity, Feature, FeatureView, FileSource, ValueType

# Entity: Stock symbol + timestamp
symbol_entity = Entity(
    name="symbol",
    value_type=ValueType.STRING,
    description="Stock symbol (e.g., AAPL, MSFT)",
)

# Data source: Market data from TimescaleDB (exported to Parquet for offline)
market_data_source = FileSource(
    path="data/features/market_data.parquet",
    event_timestamp_column="timestamp",
)

# Feature View: Technical Indicators
technical_features = FeatureView(
    name="technical_features_v1",
    entities=["symbol"],
    ttl=timedelta(days=1),
    features=[
        Feature(name="nma_9", dtype=ValueType.FLOAT),
        Feature(name="nma_20", dtype=ValueType.FLOAT),
        Feature(name="nma_50", dtype=ValueType.FLOAT),
        Feature(name="nma_200", dtype=ValueType.FLOAT),
        Feature(name="ema_9", dtype=ValueType.FLOAT),
        Feature(name="ema_20", dtype=ValueType.FLOAT),
        Feature(name="bb_pct", dtype=ValueType.FLOAT),
        Feature(name="bb_width", dtype=ValueType.FLOAT),
        Feature(name="volume_z_20", dtype=ValueType.FLOAT),
        Feature(name="volume_z_50", dtype=ValueType.FLOAT),
        Feature(name="rsi_14", dtype=ValueType.FLOAT),
        Feature(name="rsi_21", dtype=ValueType.FLOAT),
        Feature(name="macd", dtype=ValueType.FLOAT),
        Feature(name="macd_signal", dtype=ValueType.FLOAT),
        Feature(name="macd_hist", dtype=ValueType.FLOAT),
        Feature(name="atr_14", dtype=ValueType.FLOAT),
        Feature(name="stoch_k", dtype=ValueType.FLOAT),
        Feature(name="stoch_d", dtype=ValueType.FLOAT),
        Feature(name="obv", dtype=ValueType.FLOAT),
        Feature(name="obv_ema", dtype=ValueType.FLOAT),
        Feature(name="roc_10", dtype=ValueType.FLOAT),
        Feature(name="roc_20", dtype=ValueType.FLOAT),
        Feature(name="adx", dtype=ValueType.FLOAT),
        Feature(name="return_1d", dtype=ValueType.FLOAT),
        Feature(name="return_5d", dtype=ValueType.FLOAT),
        Feature(name="return_10d", dtype=ValueType.FLOAT),
        Feature(name="return_20d", dtype=ValueType.FLOAT),
        Feature(name="volatility_20d", dtype=ValueType.FLOAT),
        Feature(name="volatility_50d", dtype=ValueType.FLOAT),
    ],
    online=True,
    source=market_data_source,
    tags={"version": "v1.0.0", "owner": "ml-team"},
)

# Additional feature views can be added:
# - sentiment_features
# - microstructure_features
# - regime_features
# - macro_features
