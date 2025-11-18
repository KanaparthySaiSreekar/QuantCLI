"""
Unit tests for Signal Generator module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from src.signals.generator import (
    Signal,
    SignalType,
    SignalGenerator,
    BatchSignalGenerator
)
from src.core.exceptions import ValidationError, ModelError


class TestSignal:
    """Test suite for Signal dataclass."""

    def test_valid_signal_creation(self):
        """Test creating a valid signal."""
        signal = Signal(
            symbol='AAPL',
            timestamp=datetime.now(),
            signal_type=SignalType.BUY,
            strength=0.75,
            confidence=0.80,
            metadata={'test': True}
        )

        assert signal.symbol == 'AAPL'
        assert signal.signal_type == SignalType.BUY
        assert signal.strength == 0.75
        assert signal.confidence == 0.80

    def test_invalid_strength_raises_error(self):
        """Test that invalid strength raises ValidationError."""
        with pytest.raises(ValidationError, match="Signal strength must be 0-1"):
            Signal(
                symbol='AAPL',
                timestamp=datetime.now(),
                signal_type=SignalType.BUY,
                strength=1.5,  # Invalid: > 1.0
                confidence=0.80,
                metadata={}
            )

        with pytest.raises(ValidationError, match="Signal strength must be 0-1"):
            Signal(
                symbol='AAPL',
                timestamp=datetime.now(),
                signal_type=SignalType.SELL,
                strength=-0.5,  # Invalid: < 0.0
                confidence=0.80,
                metadata={}
            )

    def test_invalid_confidence_raises_error(self):
        """Test that invalid confidence raises ValidationError."""
        with pytest.raises(ValidationError, match="Confidence must be 0-1"):
            Signal(
                symbol='AAPL',
                timestamp=datetime.now(),
                signal_type=SignalType.BUY,
                strength=0.75,
                confidence=1.2,  # Invalid: > 1.0
                metadata={}
            )


class TestSignalGenerator:
    """Test suite for SignalGenerator."""

    def test_init_valid_params(self):
        """Test initialization with valid parameters."""
        gen = SignalGenerator(symbol='AAPL', min_confidence=0.6, min_strength=0.5)

        assert gen.symbol == 'AAPL'
        assert gen.min_confidence == 0.6
        assert gen.min_strength == 0.5

    def test_init_invalid_confidence(self):
        """Test initialization with invalid confidence threshold."""
        with pytest.raises(ValidationError, match="min_confidence must be 0-1"):
            SignalGenerator(symbol='AAPL', min_confidence=1.5)

    def test_init_invalid_strength(self):
        """Test initialization with invalid strength threshold."""
        with pytest.raises(ValidationError, match="min_strength must be 0-1"):
            SignalGenerator(symbol='AAPL', min_strength=-0.1)

    def test_generate_hold_signal_no_predictions(self, sample_ohlcv_data):
        """Test that HOLD signal is generated when no predictions provided."""
        gen = SignalGenerator(symbol='AAPL')
        signal = gen.generate(market_data=sample_ohlcv_data)

        assert signal.signal_type == SignalType.HOLD
        assert signal.strength == 0.0
        assert signal.confidence == 0.0

    def test_generate_buy_signal(self, sample_ohlcv_data):
        """Test generating BUY signal with positive prediction."""
        gen = SignalGenerator(symbol='AAPL', min_confidence=0.5, min_strength=0.4)

        predictions = {
            'direction': 1,  # Buy
            'confidence': 0.75
        }

        signal = gen.generate(
            market_data=sample_ohlcv_data,
            model_predictions=predictions
        )

        assert signal.signal_type == SignalType.BUY
        assert signal.strength > 0
        assert signal.confidence == 0.75
        assert signal.symbol == 'AAPL'

    def test_generate_sell_signal(self, sample_ohlcv_data):
        """Test generating SELL signal with negative prediction."""
        gen = SignalGenerator(symbol='AAPL', min_confidence=0.5, min_strength=0.4)

        predictions = {
            'direction': -1,  # Sell
            'confidence': 0.80
        }

        signal = gen.generate(
            market_data=sample_ohlcv_data,
            model_predictions=predictions
        )

        assert signal.signal_type == SignalType.SELL
        assert signal.strength > 0
        assert signal.confidence == 0.80

    def test_filter_low_confidence_signal(self, sample_ohlcv_data):
        """Test that low confidence signals are filtered to HOLD."""
        gen = SignalGenerator(symbol='AAPL', min_confidence=0.7, min_strength=0.5)

        predictions = {
            'direction': 1,
            'confidence': 0.4  # Below threshold
        }

        signal = gen.generate(
            market_data=sample_ohlcv_data,
            model_predictions=predictions
        )

        # Should be filtered to HOLD
        assert signal.signal_type == SignalType.HOLD

    def test_validate_market_data_missing_columns(self):
        """Test that missing columns raise ValidationError."""
        gen = SignalGenerator(symbol='AAPL')

        # Create incomplete data (missing 'volume')
        df = pd.DataFrame({
            'open': [100, 101],
            'high': [102, 103],
            'low': [99, 100],
            'close': [101, 102]
        })

        with pytest.raises(ValidationError, match="missing columns"):
            gen.generate(market_data=df)

    def test_validate_market_data_empty(self):
        """Test that empty market data raises ValidationError."""
        gen = SignalGenerator(symbol='AAPL')

        df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

        with pytest.raises(ValidationError, match="empty"):
            gen.generate(market_data=df)

    def test_strength_calculation_with_volatility(self, sample_ohlcv_data):
        """Test that strength is adjusted for volatility."""
        gen = SignalGenerator(symbol='AAPL', min_confidence=0.3, min_strength=0.1)

        predictions = {
            'direction': 1,
            'confidence': 0.8
        }

        signal = gen.generate(
            market_data=sample_ohlcv_data,
            model_predictions=predictions
        )

        # Strength should be adjusted based on volatility
        assert 0.0 < signal.strength <= 1.0

    def test_signal_metadata(self, sample_ohlcv_data, sample_features):
        """Test that signal includes proper metadata."""
        gen = SignalGenerator(symbol='AAPL')

        predictions = {
            'direction': 1,
            'confidence': 0.75
        }

        signal = gen.generate(
            market_data=sample_ohlcv_data,
            features=sample_features,
            model_predictions=predictions
        )

        assert 'model_direction' in signal.metadata
        assert 'raw_confidence' in signal.metadata
        assert 'feature_count' in signal.metadata
        assert signal.metadata['feature_count'] == len(sample_features.columns)


class TestBatchSignalGenerator:
    """Test suite for BatchSignalGenerator."""

    def test_init(self, sample_symbol_list):
        """Test BatchSignalGenerator initialization."""
        batch_gen = BatchSignalGenerator(
            symbols=sample_symbol_list,
            min_confidence=0.6,
            min_strength=0.5
        )

        assert len(batch_gen.generators) == len(sample_symbol_list)
        assert all(sym in batch_gen.generators for sym in sample_symbol_list)

    def test_generate_batch(self, sample_symbol_list, sample_ohlcv_data):
        """Test batch signal generation."""
        batch_gen = BatchSignalGenerator(
            symbols=sample_symbol_list,
            min_confidence=0.5,
            min_strength=0.4
        )

        # Create market data for all symbols
        market_data = {sym: sample_ohlcv_data for sym in sample_symbol_list}

        # Create predictions for all symbols
        predictions = {
            sym: {'direction': 1 if i % 2 == 0 else -1, 'confidence': 0.75}
            for i, sym in enumerate(sample_symbol_list)
        }

        signals = batch_gen.generate_batch(
            market_data=market_data,
            predictions=predictions
        )

        assert len(signals) == len(sample_symbol_list)
        assert all(isinstance(s, Signal) for s in signals)

    def test_generate_batch_missing_data(self, sample_symbol_list, sample_ohlcv_data):
        """Test batch generation with missing data for some symbols."""
        batch_gen = BatchSignalGenerator(symbols=sample_symbol_list)

        # Only provide data for first 3 symbols
        market_data = {
            sym: sample_ohlcv_data
            for sym in sample_symbol_list[:3]
        }

        predictions = {
            sym: {'direction': 1, 'confidence': 0.75}
            for sym in sample_symbol_list
        }

        signals = batch_gen.generate_batch(
            market_data=market_data,
            predictions=predictions
        )

        # Should only get signals for symbols with data
        assert len(signals) == 3

    def test_generate_batch_empty_market_data(self, sample_symbol_list):
        """Test that empty market_data raises ValidationError."""
        batch_gen = BatchSignalGenerator(symbols=sample_symbol_list)

        with pytest.raises(ValidationError, match="cannot be empty"):
            batch_gen.generate_batch(market_data={}, predictions={})

    def test_rank_signals(self, sample_symbol_list):
        """Test signal ranking by strength and confidence."""
        batch_gen = BatchSignalGenerator(symbols=sample_symbol_list)

        # Create signals with varying strength/confidence
        signals = [
            Signal('AAPL', datetime.now(), SignalType.BUY, 0.8, 0.9, {}),
            Signal('GOOGL', datetime.now(), SignalType.BUY, 0.6, 0.7, {}),
            Signal('MSFT', datetime.now(), SignalType.SELL, 0.9, 0.8, {}),
            Signal('AMZN', datetime.now(), SignalType.HOLD, 0.0, 0.0, {}),  # Should be filtered
            Signal('TSLA', datetime.now(), SignalType.BUY, 0.5, 0.6, {})
        ]

        ranked = batch_gen.rank_signals(signals)

        # Should exclude HOLD signals
        assert len(ranked) == 4

        # Should be sorted by score (descending)
        scores = [score for _, score in ranked]
        assert scores == sorted(scores, reverse=True)

        # Top signal should be AAPL (highest score)
        assert ranked[0][0].symbol == 'AAPL'

    def test_rank_signals_with_top_n(self, sample_symbol_list):
        """Test ranking with top_n limit."""
        batch_gen = BatchSignalGenerator(symbols=sample_symbol_list)

        signals = [
            Signal(sym, datetime.now(), SignalType.BUY, 0.7, 0.8, {})
            for sym in sample_symbol_list
        ]

        ranked = batch_gen.rank_signals(signals, top_n=3)

        assert len(ranked) == 3

    def test_batch_error_handling(self, sample_symbol_list, sample_ohlcv_data):
        """Test that errors in individual symbols don't crash batch."""
        batch_gen = BatchSignalGenerator(symbols=sample_symbol_list)

        market_data = {sym: sample_ohlcv_data for sym in sample_symbol_list}

        # Predictions with invalid data for one symbol
        predictions = {
            'AAPL': {'direction': 1, 'confidence': 0.75},
            'GOOGL': {'direction': 1, 'confidence': 0.75},
            # Missing predictions for MSFT will cause it to be skipped
            'AMZN': {'direction': 1, 'confidence': 0.75},
            'TSLA': {'direction': 1, 'confidence': 0.75}
        }

        signals = batch_gen.generate_batch(
            market_data=market_data,
            predictions=predictions
        )

        # Should get signals for all symbols (missing predictions = HOLD)
        assert len(signals) == len(sample_symbol_list)
