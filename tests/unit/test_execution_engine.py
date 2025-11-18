"""
Unit tests for execution engine.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock

from src.execution.execution_engine import ExecutionEngine
from src.execution.broker import IBKRClient
from src.execution.order_manager import OrderManager
from src.execution.position_manager import PositionManager
from src.signals.generator import Signal, SignalType


class TestExecutionEngine:
    """Tests for ExecutionEngine."""

    @pytest.fixture
    def mock_broker(self):
        """Mock broker client."""
        broker = Mock(spec=IBKRClient)
        broker.place_order.return_value = 1
        return broker

    @pytest.fixture
    def engine(self, mock_broker):
        """Create execution engine."""
        return ExecutionEngine(
            broker=mock_broker,
            max_position_size=0.10,
            max_portfolio_exposure=1.0
        )

    def test_execute_buy_signal(self, engine, mock_broker):
        """Test executing buy signal."""
        signal = Signal(
            symbol='AAPL',
            timestamp=datetime.now(),
            signal_type=SignalType.BUY,
            strength=0.8,
            confidence=0.9,
            metadata={}
        )

        order_id = engine.execute_signal(signal, current_price=150.0)

        assert order_id is not None
        mock_broker.place_order.assert_called_once()

        # Check position was created
        position = engine.position_manager.get_position('AAPL')
        assert position is not None
        assert position.quantity > 0

    def test_execute_sell_signal(self, engine, mock_broker):
        """Test executing sell signal."""
        # First create a position
        engine.position_manager.open_position('AAPL', 100, 150.0)

        signal = Signal(
            symbol='AAPL',
            timestamp=datetime.now(),
            signal_type=SignalType.SELL,
            strength=0.7,
            confidence=0.8,
            metadata={}
        )

        order_id = engine.execute_signal(signal, current_price=155.0)

        assert order_id is not None
        mock_broker.place_order.assert_called()

    def test_hold_signal_no_action(self, engine, mock_broker):
        """Test that HOLD signal doesn't place orders."""
        signal = Signal(
            symbol='AAPL',
            timestamp=datetime.now(),
            signal_type=SignalType.HOLD,
            strength=0.0,
            confidence=0.0,
            metadata={}
        )

        order_id = engine.execute_signal(signal, current_price=150.0)

        assert order_id is None
        mock_broker.place_order.assert_not_called()

    def test_position_size_calculation(self, engine):
        """Test position size calculation."""
        signal = Signal(
            symbol='AAPL',
            timestamp=datetime.now(),
            signal_type=SignalType.BUY,
            strength=0.8,
            confidence=0.9,
            metadata={}
        )

        size = engine._calculate_position_size(signal, current_price=100.0)

        # Should be based on portfolio value, max position size, and signal strength
        expected_value = engine.portfolio_value * engine.max_position_size * (0.8 * 0.9)
        expected_shares = int(expected_value / 100.0)

        assert size == expected_shares

    def test_risk_check_position_size_limit(self, engine):
        """Test risk check fails when position too large."""
        from src.core.exceptions import RiskError

        # Try to place order larger than limit
        large_quantity = int(engine.portfolio_value * 0.15 / 100.0)  # 15% of portfolio

        with pytest.raises(RiskError, match="exceeds limit"):
            engine._pre_trade_risk_check('AAPL', 'BUY', large_quantity, 100.0)

    def test_batch_signal_execution(self, engine, mock_broker, sample_signal):
        """Test executing multiple signals."""
        signals = [
            Signal('AAPL', datetime.now(), SignalType.BUY, 0.8, 0.9, {}),
            Signal('GOOGL', datetime.now(), SignalType.BUY, 0.7, 0.8, {}),
            Signal('MSFT', datetime.now(), SignalType.SELL, 0.6, 0.7, {})
        ]

        prices = {'AAPL': 150.0, 'GOOGL': 2800.0, 'MSFT': 350.0}

        results = engine.execute_batch_signals(signals, prices)

        assert len(results) == 3
        assert all(symbol in results for symbol in ['AAPL', 'GOOGL', 'MSFT'])

    def test_portfolio_summary(self, engine):
        """Test portfolio summary generation."""
        # Create some positions
        engine.position_manager.open_position('AAPL', 100, 150.0)
        engine.position_manager.open_position('GOOGL', 10, 2800.0)

        # Update prices
        engine.position_manager.update_prices({
            'AAPL': 155.0,
            'GOOGL': 2850.0
        })

        summary = engine.get_portfolio_summary()

        assert 'portfolio_value' in summary
        assert 'n_positions' in summary
        assert summary['n_positions'] == 2
        assert 'total_pnl' in summary
        assert 'positions' in summary
        assert len(summary['positions']) == 2
