"""Tests for feature engineering."""

import pytest
import pandas as pd
import numpy as np

from src.features.technical import TechnicalIndicators
from src.features.engineer import FeatureEngineer


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
