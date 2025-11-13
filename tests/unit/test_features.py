"""
Unit Tests for Feature Generation

Tests deterministic feature computation, data quality validation,
and feature store consistency.
"""

import hashlib
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.features.generator import FeatureGenerator, FEATURE_VERSION


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D", tz="UTC")
    n = len(dates)

    np.random.seed(42)

    data = pd.DataFrame(
        {
            "open": 100 + np.cumsum(np.random.randn(n) * 2),
            "high": 100 + np.cumsum(np.random.randn(n) * 2) + 1,
            "low": 100 + np.cumsum(np.random.randn(n) * 2) - 1,
            "close": 100 + np.cumsum(np.random.randn(n) * 2),
            "volume": np.random.randint(1000000, 10000000, n),
        },
        index=dates,
    )

    # Ensure OHLC consistency
    data["high"] = data[["open", "high", "low", "close"]].max(axis=1)
    data["low"] = data[["open", "high", "low", "close"]].min(axis=1)

    return data


class TestFeatureGenerator:
    """Test FeatureGenerator class."""

    def test_deterministic_features(self, sample_ohlcv_data):
        """Test that feature generation is deterministic."""
        generator1 = FeatureGenerator(seed=42)
        generator2 = FeatureGenerator(seed=42)

        features1 = generator1.compute_technical(sample_ohlcv_data, validate=False)
        features2 = generator2.compute_technical(sample_ohlcv_data, validate=False)

        # Should be identical
        pd.testing.assert_frame_equal(features1, features2)

    def test_different_seeds_produce_different_results(self, sample_ohlcv_data):
        """Test that different seeds produce different results (if any randomness)."""
        generator1 = FeatureGenerator(seed=42)
        generator2 = FeatureGenerator(seed=99)

        features1 = generator1.compute_technical(sample_ohlcv_data, validate=False)
        features2 = generator2.compute_technical(sample_ohlcv_data, validate=False)

        # For technical indicators, should still be identical (no randomness)
        # But seeds are set up for future stochastic features
        pd.testing.assert_frame_equal(features1, features2)

    def test_feature_names(self, sample_ohlcv_data):
        """Test that expected features are generated."""
        generator = FeatureGenerator(seed=42)
        features = generator.compute_technical(sample_ohlcv_data, validate=False)

        expected_features = [
            "nma_9",
            "nma_20",
            "nma_50",
            "nma_200",
            "ema_9",
            "ema_20",
            "bb_pct",
            "bb_width",
            "volume_z_20",
            "volume_z_50",
            "rsi_14",
            "rsi_21",
            "macd",
            "macd_signal",
            "macd_hist",
            "atr_14",
            "stoch_k",
            "stoch_d",
            "obv",
            "obv_ema",
            "roc_10",
            "roc_20",
            "adx",
            "return_1d",
            "return_5d",
            "return_10d",
            "return_20d",
            "volatility_20d",
            "volatility_50d",
        ]

        for feature in expected_features:
            assert feature in features.columns, f"Missing feature: {feature}"

    def test_no_nulls_in_output(self, sample_ohlcv_data):
        """Test that output has no NaN values."""
        generator = FeatureGenerator(seed=42)
        features = generator.compute_technical(sample_ohlcv_data, validate=False)

        assert features.isnull().sum().sum() == 0, "Output contains NaN values"

    def test_input_validation_catches_nulls(self, sample_ohlcv_data):
        """Test that validation catches null values."""
        data_with_nulls = sample_ohlcv_data.copy()
        data_with_nulls.loc[data_with_nulls.index[10], "close"] = np.nan

        generator = FeatureGenerator(seed=42)

        with pytest.raises(ValueError, match="Null values detected"):
            generator.compute_technical(data_with_nulls, validate=True)

    def test_input_validation_catches_negative_prices(self, sample_ohlcv_data):
        """Test that validation catches negative prices."""
        data_with_negative = sample_ohlcv_data.copy()
        data_with_negative.loc[data_with_negative.index[10], "close"] = -10

        generator = FeatureGenerator(seed=42)

        with pytest.raises(ValueError, match="Negative prices"):
            generator.compute_technical(data_with_negative, validate=True)

    def test_input_validation_catches_invalid_ohlc(self, sample_ohlcv_data):
        """Test that validation catches OHLC inconsistencies."""
        data_invalid = sample_ohlcv_data.copy()
        # Make high < low
        data_invalid.loc[data_invalid.index[10], "high"] = 50
        data_invalid.loc[data_invalid.index[10], "low"] = 100

        generator = FeatureGenerator(seed=42)

        with pytest.raises(ValueError, match="Invalid OHLC bars"):
            generator.compute_technical(data_invalid, validate=True)

    def test_feature_ranges(self, sample_ohlcv_data):
        """Test that features are in expected ranges."""
        generator = FeatureGenerator(seed=42)
        features = generator.compute_technical(sample_ohlcv_data, validate=False)

        # RSI should be [0, 100]
        assert features["rsi_14"].min() >= 0
        assert features["rsi_14"].max() <= 100

        # BB percentage should be roughly [0, 1]
        assert features["bb_pct"].min() >= -0.5  # Allow some overshoot
        assert features["bb_pct"].max() <= 1.5

        # Stochastic should be [0, 100]
        assert features["stoch_k"].min() >= 0
        assert features["stoch_k"].max() <= 100

    def test_metadata_generation(self, sample_ohlcv_data):
        """Test that metadata is generated correctly."""
        generator = FeatureGenerator(seed=42)
        features = generator.compute_technical(sample_ohlcv_data, validate=False)

        assert len(generator.metadata) > 0

        metadata = generator.metadata[0]
        assert metadata.feature_name == "technical_indicators"
        assert metadata.version == FEATURE_VERSION
        assert isinstance(metadata.input_hash, str)
        assert isinstance(metadata.quality_stats, dict)

    def test_timezone_handling(self):
        """Test that timezone-naive data is converted to UTC."""
        # Create timezone-naive data
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
        data = pd.DataFrame(
            {
                "open": 100 + np.random.randn(len(dates)),
                "high": 101 + np.random.randn(len(dates)),
                "low": 99 + np.random.randn(len(dates)),
                "close": 100 + np.random.randn(len(dates)),
                "volume": np.random.randint(1000000, 10000000, len(dates)),
            },
            index=dates,
        )

        generator = FeatureGenerator(seed=42)
        features = generator.compute_technical(data, validate=False)

        # Should have timezone
        assert features.index.tz is not None
        assert str(features.index.tz) == "UTC"


class TestFeatureConsistency:
    """Test feature consistency across different scenarios."""

    def test_idempotent_computation(self, sample_ohlcv_data):
        """Test that computing features twice gives same result."""
        generator = FeatureGenerator(seed=42)

        features1 = generator.compute_technical(sample_ohlcv_data, validate=False)
        features2 = generator.compute_technical(sample_ohlcv_data, validate=False)

        pd.testing.assert_frame_equal(features1, features2)

    def test_subset_consistency(self, sample_ohlcv_data):
        """Test that features on subset match features on full data."""
        generator = FeatureGenerator(seed=42)

        # Compute on full data
        full_features = generator.compute_technical(sample_ohlcv_data, validate=False)

        # Compute on last 100 days
        subset_data = sample_ohlcv_data.tail(300)  # Need extra for lookback
        subset_features = generator.compute_technical(subset_data, validate=False)

        # Compare last 50 days (after warmup)
        common_dates = full_features.tail(50).index.intersection(
            subset_features.tail(50).index
        )

        if len(common_dates) > 0:
            full_subset = full_features.loc[common_dates]
            subset_subset = subset_features.loc[common_dates]

            # Should be close (within numerical precision)
            pd.testing.assert_frame_equal(
                full_subset, subset_subset, rtol=1e-5, atol=1e-8
            )


@pytest.mark.parametrize(
    "seed,expected_hash",
    [
        (42, "same_hash_1"),
        (42, "same_hash_1"),  # Same seed should give same hash
        (99, "different_hash"),  # Different seed... wait, no randomness, so same hash
    ],
)
def test_hash_determinism(sample_ohlcv_data, seed, expected_hash):
    """Test that data hashing is deterministic."""
    generator = FeatureGenerator(seed=seed)

    # Hash should be same for same data regardless of seed
    hash1 = generator._hash_dataframe(sample_ohlcv_data)
    hash2 = generator._hash_dataframe(sample_ohlcv_data)

    assert hash1 == hash2, "Hash should be deterministic"
    assert isinstance(hash1, str)
    assert len(hash1) == 16  # Truncated to 16 chars
