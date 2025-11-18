"""
Feature Store - Single Source of Truth for Features

This module provides deterministic, reproducible feature computation
with versioning, metadata tracking, and identical offline/online serving.

Design Principles:
- Idempotent: Running twice produces identical results
- Deterministic: Fixed seeds, sorted operations
- Versioned: Every feature has a version
- Validated: Data quality checks on every computation
"""

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

# Feature versioning
FEATURE_VERSION = "v1.0.0"


class FeatureMetadata:
    """Metadata for feature computation tracking."""

    def __init__(
        self,
        feature_name: str,
        version: str,
        data_start: datetime,
        data_end: datetime,
        input_hash: str,
        quality_stats: Dict,
    ):
        self.feature_name = feature_name
        self.version = version
        self.data_start = data_start
        self.data_end = data_end
        self.input_hash = input_hash
        self.quality_stats = quality_stats
        self.computed_at = datetime.utcnow()

    def to_dict(self) -> Dict:
        return {
            "feature_name": self.feature_name,
            "version": self.version,
            "data_start": self.data_start.isoformat(),
            "data_end": self.data_end.isoformat(),
            "input_hash": self.input_hash,
            "quality_stats": self.quality_stats,
            "computed_at": self.computed_at.isoformat(),
        }


class FeatureGenerator:
    """
    Deterministic feature generator with quality validation.

    Guarantees:
    - Same input → same output (deterministic)
    - Timezone-aware (UTC)
    - Fixed random seeds
    - Idempotent windowing
    """

    def __init__(self, seed: int = 42):
        """
        Initialize feature generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
        self.metadata: List[FeatureMetadata] = []

    def _hash_dataframe(self, df: pd.DataFrame) -> str:
        """Compute deterministic hash of input data."""
        # Sort by index to ensure deterministic ordering
        df_sorted = df.sort_index()
        content = df_sorted.to_json(orient="split", date_format="iso")
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _validate_input(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate input data quality.

        Returns:
            (is_valid, list_of_issues)
        """
        issues = []

        # Check for required columns
        required = ["open", "high", "low", "close", "volume"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            issues.append(f"Missing columns: {missing}")

        # Check for nulls
        null_cols = df[required].isnull().sum()
        if null_cols.any():
            issues.append(f"Null values detected: {null_cols[null_cols > 0].to_dict()}")

        # Check for negative prices
        if (df[["open", "high", "low", "close"]] < 0).any().any():
            issues.append("Negative prices detected")

        # Check for zero volume
        if (df["volume"] == 0).sum() > len(df) * 0.1:  # >10% zero volume
            issues.append("Excessive zero volume bars")

        # Check for OHLC consistency
        invalid_bars = (
            (df["high"] < df["low"])
            | (df["close"] > df["high"])
            | (df["close"] < df["low"])
            | (df["open"] > df["high"])
            | (df["open"] < df["low"])
        ).sum()
        if invalid_bars > 0:
            issues.append(f"Invalid OHLC bars: {invalid_bars}")

        return len(issues) == 0, issues

    def _compute_quality_stats(self, df: pd.DataFrame) -> Dict:
        """Compute data quality statistics."""
        return {
            "n_rows": len(df),
            "n_nulls": df.isnull().sum().sum(),
            "date_range_days": (df.index[-1] - df.index[0]).days,
            "mean_volume": float(df["volume"].mean()),
            "std_volume": float(df["volume"].std()),
            "mean_close": float(df["close"].mean()),
            "std_close": float(df["close"].std()),
        }

    def compute_technical(
        self, df: pd.DataFrame, validate: bool = True
    ) -> pd.DataFrame:
        """
        Compute technical indicators (deterministic).

        Args:
            df: OHLCV DataFrame with DatetimeIndex (UTC)
            validate: Whether to validate input data

        Returns:
            DataFrame with technical features

        Raises:
            ValueError: If validation fails
        """
        logger.info(f"Computing technical features on {len(df)} bars")

        # Ensure UTC timezone
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        # Validate input
        if validate:
            is_valid, issues = self._validate_input(df)
            if not is_valid:
                raise ValueError(f"Input validation failed: {issues}")

        # Create copy to avoid modifying original
        result = df.copy()

        # Hash input for tracking
        input_hash = self._hash_dataframe(df)

        # Normalized Moving Averages (NMA)
        # Research: 9% R² improvement
        for period in [9, 20, 50, 200]:
            sma = result["close"].rolling(window=period, min_periods=period).mean()
            result[f"nma_{period}"] = (result["close"] - sma) / sma

        # Exponential Moving Averages
        for period in [9, 20]:
            ema = result["close"].ewm(span=period, adjust=False, min_periods=period).mean()
            result[f"ema_{period}"] = ema

        # Bollinger Bands Percentage
        # Research: 14.7% feature importance
        window = 20
        sma = result["close"].rolling(window=window, min_periods=window).mean()
        std = result["close"].rolling(window=window, min_periods=window).std()
        upper = sma + (2 * std)
        lower = sma - (2 * std)
        result["bb_pct"] = (result["close"] - lower) / (upper - lower)
        result["bb_width"] = (upper - lower) / sma

        # Volume Z-Score
        # Research: 14-17% feature importance
        for period in [20, 50]:
            vol_mean = result["volume"].rolling(window=period, min_periods=period).mean()
            vol_std = result["volume"].rolling(window=period, min_periods=period).std()
            result[f"volume_z_{period}"] = (result["volume"] - vol_mean) / (
                vol_std + 1e-8
            )

        # RSI (Relative Strength Index)
        for period in [14, 21]:
            delta = result["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period).mean()
            rs = gain / (loss + 1e-8)
            result[f"rsi_{period}"] = 100 - (100 / (1 + rs))

        # MACD (Moving Average Convergence Divergence)
        exp1 = result["close"].ewm(span=12, adjust=False, min_periods=12).mean()
        exp2 = result["close"].ewm(span=26, adjust=False, min_periods=26).mean()
        result["macd"] = exp1 - exp2
        result["macd_signal"] = result["macd"].ewm(span=9, adjust=False, min_periods=9).mean()
        result["macd_hist"] = result["macd"] - result["macd_signal"]

        # ATR (Average True Range) - volatility
        high_low = result["high"] - result["low"]
        high_close = abs(result["high"] - result["close"].shift())
        low_close = abs(result["low"] - result["close"].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        result["atr_14"] = true_range.rolling(window=14, min_periods=14).mean()

        # Stochastic Oscillator
        period = 14
        low_min = result["low"].rolling(window=period, min_periods=period).min()
        high_max = result["high"].rolling(window=period, min_periods=period).max()
        result["stoch_k"] = 100 * (result["close"] - low_min) / (high_max - low_min + 1e-8)
        result["stoch_d"] = result["stoch_k"].rolling(window=3, min_periods=3).mean()

        # OBV (On-Balance Volume)
        obv = (
            np.sign(result["close"].diff())
            .fillna(0)
            .mul(result["volume"])
            .cumsum()
        )
        result["obv"] = obv
        result["obv_ema"] = obv.ewm(span=20, adjust=False, min_periods=20).mean()

        # Rate of Change
        for period in [10, 20]:
            result[f"roc_{period}"] = (
                result["close"].pct_change(periods=period) * 100
            )

        # ADX (Average Directional Index) - trend strength
        plus_dm = result["high"].diff()
        minus_dm = -result["low"].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        atr = true_range.rolling(window=14, min_periods=14).mean()
        plus_di = 100 * (plus_dm.rolling(window=14, min_periods=14).mean() / (atr + 1e-8))
        minus_di = 100 * (minus_dm.rolling(window=14, min_periods=14).mean() / (atr + 1e-8))

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        result["adx"] = dx.rolling(window=14, min_periods=14).mean()

        # Returns (various horizons)
        for period in [1, 5, 10, 20]:
            result[f"return_{period}d"] = result["close"].pct_change(periods=period)

        # Volatility (rolling std of returns)
        returns = result["close"].pct_change()
        for period in [20, 50]:
            result[f"volatility_{period}d"] = returns.rolling(window=period, min_periods=period).std()

        # Drop NaN rows (from indicators with lookback)
        initial_len = len(result)
        result = result.dropna()
        logger.info(f"Dropped {initial_len - len(result)} rows with NaN values")

        # Store metadata
        quality_stats = self._compute_quality_stats(result)
        metadata = FeatureMetadata(
            feature_name="technical_indicators",
            version=FEATURE_VERSION,
            data_start=result.index[0].to_pydatetime(),
            data_end=result.index[-1].to_pydatetime(),
            input_hash=input_hash,
            quality_stats=quality_stats,
        )
        self.metadata.append(metadata)

        logger.success(
            f"Computed {len([c for c in result.columns if c not in df.columns])} technical features"
        )

        return result

    def compute_all_features(
        self,
        symbol: str,
        df: pd.DataFrame,
        include_sentiment: bool = False,
        include_microstructure: bool = False,
    ) -> pd.DataFrame:
        """
        Compute all features (modular approach).

        Args:
            symbol: Stock symbol
            df: OHLCV DataFrame
            include_sentiment: Whether to include sentiment features
            include_microstructure: Whether to include microstructure features

        Returns:
            DataFrame with all features
        """
        logger.info(f"Building feature set for {symbol}")

        # Start with technical features
        features = self.compute_technical(df)

        # Add sentiment features (if available)
        if include_sentiment:
            # TODO: Implement sentiment feature integration
            logger.info("Sentiment features not yet implemented")

        # Add microstructure features (if available)
        if include_microstructure:
            # TODO: Implement microstructure feature integration
            logger.info("Microstructure features not yet implemented")

        # Add symbol as feature (for cross-sectional models)
        features["symbol"] = symbol

        return features

    def save_metadata(self, path: Path):
        """Save feature metadata to JSON."""
        metadata_dicts = [m.to_dict() for m in self.metadata]
        with open(path, "w") as f:
            json.dump(metadata_dicts, f, indent=2)
        logger.success(f"Saved feature metadata to {path}")

    def load_metadata(self, path: Path):
        """Load feature metadata from JSON."""
        with open(path, "r") as f:
            metadata_dicts = json.load(f)
        # TODO: Reconstruct FeatureMetadata objects
        logger.success(f"Loaded {len(metadata_dicts)} metadata records from {path}")


# Global feature generator instance (singleton)
_feature_generator = None


def get_feature_generator(seed: int = 42) -> FeatureGenerator:
    """Get global feature generator instance."""
    global _feature_generator
    if _feature_generator is None:
        _feature_generator = FeatureGenerator(seed=seed)
    return _feature_generator
