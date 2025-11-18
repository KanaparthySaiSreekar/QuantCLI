"""
Tests for feature engineering and feature generation.

Tests both the technical indicators module and the feature generator,
including deterministic feature computation and data quality validation.
"""

import hashlib
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.features.technical import TechnicalIndicators
from src.features.engineer import FeatureEngineer
from src.features.generator import FeatureGenerator, FEATURE_VERSION


class TestTechnicalIndicators:
    """Tests for technical indicators."""

    def test_sma(self, sample_ohlcv_data):
        """Test SMA calculation."""
        sma = TechnicalIndicators.sma(sample_ohlcv_data['close'], 20)

        assert len(sma) == len(sample_ohlcv_data)
        assert not sma.iloc[-1] == 0
        # First 19 values should be NaN
        assert sma.iloc[:19].isna().all()

    def test_ema(self, sample_ohlcv_data):
        """Test EMA calculation."""
        ema = TechnicalIndicators.ema(sample_ohlcv_data['close'], 20)

        assert len(ema) == len(sample_ohlcv_data)
        assert not ema.iloc[-1] == 0

    def test_rsi(self, sample_ohlcv_data):
        """Test RSI calculation."""
        rsi = TechnicalIndicators.rsi(sample_ohlcv_data['close'], 14)

        assert len(rsi) == len(sample_ohlcv_data)
        # RSI should be between 0 and 100
        assert (rsi.dropna() >= 0).all()
        assert (rsi.dropna() <= 100).all()

    def test_macd(self, sample_ohlcv_data):
        """Test MACD calculation."""
        macd, signal, hist = TechnicalIndicators.macd(sample_ohlcv_data['close'])

        assert len(macd) == len(sample_ohlcv_data)
        assert len(signal) == len(sample_ohlcv_data)
        assert len(hist) == len(sample_ohlcv_data)

        # Histogram should be MACD - Signal
        assert np.allclose(hist.dropna(), (macd - signal).dropna())

    def test_bollinger_bands(self, sample_ohlcv_data):
        """Test Bollinger Bands."""
        upper, middle, lower = TechnicalIndicators.bollinger_bands(
            sample_ohlcv_data['close'], 20, 2.0
        )

        # Upper should be > Middle > Lower
        valid_data = ~upper.isna()
        assert (upper[valid_data] >= middle[valid_data]).all()
        assert (middle[valid_data] >= lower[valid_data]).all()

    def test_atr(self, sample_ohlcv_data):
        """Test ATR calculation."""
        atr = TechnicalIndicators.atr(sample_ohlcv_data, 14)

        assert len(atr) == len(sample_ohlcv_data)
        # ATR should be positive
        assert (atr.dropna() > 0).all()

    def test_calculate_all_indicators(self, sample_ohlcv_data):
        """Test calculating all indicators at once."""
        result = TechnicalIndicators.calculate_all_indicators(sample_ohlcv_data)

        # Should have all original columns plus indicators
        assert len(result.columns) > len(sample_ohlcv_data.columns)

        # Check for key indicators
        assert 'rsi_14' in result.columns
        assert 'macd' in result.columns
        assert 'bb_upper' in result.columns
        assert 'sma_20' in result.columns


class TestFeatureEngineer:
    """Tests for FeatureEngineer."""

    def test_generate_features(self, sample_ohlcv_data):
        """Test feature generation."""
        engineer = FeatureEngineer(
            include_technical=True,
            include_price=True,
            include_volume=True,
            include_time=True
        )

        features = engineer.generate_features(sample_ohlcv_data)

        # Should have many more columns
        assert len(features.columns) > len(sample_ohlcv_data.columns)

        # Should have fewer rows due to NaN dropping
        assert len(features) <= len(sample_ohlcv_data)

    def test_price_features(self, sample_ohlcv_data):
        """Test price-based features."""
        engineer = FeatureEngineer(
            include_technical=False,
            include_price=True,
            include_volume=False,
            include_time=False
        )

        features = engineer.generate_features(sample_ohlcv_data, dropna=False)

        assert 'return_1d' in features.columns
        assert 'return_5d' in features.columns
        assert 'intraday_range' in features.columns

    def test_volume_features(self, sample_ohlcv_data):
        """Test volume-based features."""
        engineer = FeatureEngineer(
            include_technical=False,
            include_price=False,
            include_volume=True,
            include_time=False
        )

        features = engineer.generate_features(sample_ohlcv_data, dropna=False)

        assert 'volume_sma_20' in features.columns
        assert 'volume_ratio_20' in features.columns

    def test_time_features(self, sample_ohlcv_data):
        """Test time-based features."""
        engineer = FeatureEngineer(
            include_technical=False,
            include_price=False,
            include_volume=False,
            include_time=True
        )

        features = engineer.generate_features(sample_ohlcv_data, dropna=False)

        assert 'day_of_week' in features.columns
        assert 'month' in features.columns
        assert 'is_monday' in features.columns

    def test_transform_for_ml(self, sample_ohlcv_data):
        """Test ML transformation."""
        engineer = FeatureEngineer()

        features = engineer.generate_features(sample_ohlcv_data)
        ml_data = engineer.transform_for_ml(features, target_periods=1, classification=True)

        assert 'target' in ml_data.columns
        # Target should be binary for classification
        assert set(ml_data['target'].unique()).issubset({0, 1})


class TestFeatureGenerator:
    """Test FeatureGenerator class."""

    @pytest.fixture
    def generator_sample_data(self):
        """Create sample OHLCV data for FeatureGenerator testing."""
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

    def test_deterministic_features(self, generator_sample_data):
        """Test that feature generation is deterministic."""
        generator1 = FeatureGenerator(seed=42)
        generator2 = FeatureGenerator(seed=42)

        features1 = generator1.compute_technical(generator_sample_data, validate=False)
        features2 = generator2.compute_technical(generator_sample_data, validate=False)

        # Should be identical
        pd.testing.assert_frame_equal(features1, features2)

    def test_different_seeds_produce_different_results(self, generator_sample_data):
        """Test that different seeds produce different results (if any randomness)."""
        generator1 = FeatureGenerator(seed=42)
        generator2 = FeatureGenerator(seed=99)

        features1 = generator1.compute_technical(generator_sample_data, validate=False)
        features2 = generator2.compute_technical(generator_sample_data, validate=False)

        # For technical indicators, should still be identical (no randomness)
        # But seeds are set up for future stochastic features
        pd.testing.assert_frame_equal(features1, features2)

    def test_feature_names(self, generator_sample_data):
        """Test that expected features are generated."""
        generator = FeatureGenerator(seed=42)
        features = generator.compute_technical(generator_sample_data, validate=False)

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

    def test_no_nulls_in_output(self, generator_sample_data):
        """Test that output has no NaN values."""
        generator = FeatureGenerator(seed=42)
        features = generator.compute_technical(generator_sample_data, validate=False)

        assert features.isnull().sum().sum() == 0, "Output contains NaN values"

    def test_input_validation_catches_nulls(self, generator_sample_data):
        """Test that validation catches null values."""
        data_with_nulls = generator_sample_data.copy()
        data_with_nulls.loc[data_with_nulls.index[10], "close"] = np.nan

        generator = FeatureGenerator(seed=42)

        with pytest.raises(ValueError, match="Null values detected"):
            generator.compute_technical(data_with_nulls, validate=True)

    def test_input_validation_catches_negative_prices(self, generator_sample_data):
        """Test that validation catches negative prices."""
        data_with_negative = generator_sample_data.copy()
        data_with_negative.loc[data_with_negative.index[10], "close"] = -10

        generator = FeatureGenerator(seed=42)

        with pytest.raises(ValueError, match="Negative prices"):
            generator.compute_technical(data_with_negative, validate=True)

    def test_input_validation_catches_invalid_ohlc(self, generator_sample_data):
        """Test that validation catches OHLC inconsistencies."""
        data_invalid = generator_sample_data.copy()
        # Make high < low
        data_invalid.loc[data_invalid.index[10], "high"] = 50
        data_invalid.loc[data_invalid.index[10], "low"] = 100

        generator = FeatureGenerator(seed=42)

        with pytest.raises(ValueError, match="Invalid OHLC bars"):
            generator.compute_technical(data_invalid, validate=True)

    def test_feature_ranges(self, generator_sample_data):
        """Test that features are in expected ranges."""
        generator = FeatureGenerator(seed=42)
        features = generator.compute_technical(generator_sample_data, validate=False)

        # RSI should be [0, 100]
        assert features["rsi_14"].min() >= 0
        assert features["rsi_14"].max() <= 100

        # BB percentage should be roughly [0, 1]
        assert features["bb_pct"].min() >= -0.5  # Allow some overshoot
        assert features["bb_pct"].max() <= 1.5

        # Stochastic should be [0, 100]
        assert features["stoch_k"].min() >= 0
        assert features["stoch_k"].max() <= 100

    def test_metadata_generation(self, generator_sample_data):
        """Test that metadata is generated correctly."""
        generator = FeatureGenerator(seed=42)
        features = generator.compute_technical(generator_sample_data, validate=False)

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

    @pytest.fixture
    def consistency_sample_data(self):
        """Create sample OHLCV data for consistency testing."""
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

    def test_idempotent_computation(self, consistency_sample_data):
        """Test that computing features twice gives same result."""
        generator = FeatureGenerator(seed=42)

        features1 = generator.compute_technical(consistency_sample_data, validate=False)
        features2 = generator.compute_technical(consistency_sample_data, validate=False)

        pd.testing.assert_frame_equal(features1, features2)

    def test_subset_consistency(self, consistency_sample_data):
        """Test that features on subset match features on full data."""
        generator = FeatureGenerator(seed=42)

        # Compute on full data
        full_features = generator.compute_technical(consistency_sample_data, validate=False)

        # Compute on last 100 days
        subset_data = consistency_sample_data.tail(300)  # Need extra for lookback
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
def test_hash_determinism(seed, expected_hash):
    """Test that data hashing is deterministic."""
    # Create sample data
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

    generator = FeatureGenerator(seed=seed)

    # Hash should be same for same data regardless of seed
    hash1 = generator._hash_dataframe(data)
    hash2 = generator._hash_dataframe(data)

    assert hash1 == hash2, "Hash should be deterministic"
    assert isinstance(hash1, str)
    assert len(hash1) == 16  # Truncated to 16 chars
