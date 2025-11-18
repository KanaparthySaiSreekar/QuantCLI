"""Tests for backtesting engine."""

import pytest
import pandas as pd
import numpy as np

from src.backtest.engine import BacktestEngine, BacktestResult


class TestBacktestEngine:
    """Tests for BacktestEngine."""

    @pytest.fixture
    def engine(self):
        """Create backtest engine."""
        return BacktestEngine(
            initial_capital=100000,
            commission=0.001,
            slippage=0.0005
        )

    @pytest.fixture
    def sample_signals(self):
        """Sample trading signals."""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        signals = pd.Series([1, 1, 1, 0, -1, -1, -1, 0] * 12 + [1, 1, 1, 0], index=dates)
        return signals

    @pytest.fixture
    def sample_prices(self):
        """Sample price series."""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        # Trending up then down
        prices = pd.Series(
            [100 + i*0.5 if i < 50 else 125 - (i-50)*0.5 for i in range(100)],
            index=dates
        )
        return prices

    def test_backtest_basic(self, engine, sample_signals, sample_prices):
        """Test basic backtesting."""
        result = engine.run(sample_signals, sample_prices)

        assert isinstance(result, BacktestResult)
        assert hasattr(result, 'total_return')
        assert hasattr(result, 'sharpe_ratio')
        assert hasattr(result, 'max_drawdown')
        assert result.total_trades > 0

    def test_backtest_profitable_strategy(self, engine):
        """Test backtesting profitable strategy."""
        # Perfect trend following
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        prices = pd.Series(range(100, 150), index=dates)  # Uptrend
        signals = pd.Series([1] * 50, index=dates)  # Always long

        result = engine.run(signals, prices)

        assert result.total_return > 0
        assert result.win_rate > 50

    def test_backtest_losing_strategy(self, engine):
        """Test backtesting losing strategy."""
        # Buy high, sell low
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        prices = pd.Series(range(100, 150), index=dates)  # Uptrend
        signals = pd.Series([-1] * 50, index=dates)  # Always short (wrong)

        result = engine.run(signals, prices)

        assert result.total_return < 0

    def test_transaction_costs(self):
        """Test that transaction costs reduce returns."""
        dates = pd.date_range('2024-01-01', periods=20, freq='D')
        prices = pd.Series([100] * 20, index=dates)  # Flat
        signals = pd.Series([1, -1] * 10, index=dates)  # Lots of trading

        # With costs
        engine_with_costs = BacktestEngine(
            initial_capital=100000,
            commission=0.01,
            slippage=0.01
        )
        result_with_costs = engine_with_costs.run(signals, prices)

        # Without costs
        engine_no_costs = BacktestEngine(
            initial_capital=100000,
            commission=0.0,
            slippage=0.0
        )
        result_no_costs = engine_no_costs.run(signals, prices)

        # With costs should underperform
        assert result_with_costs.total_return < result_no_costs.total_return

    def test_sharpe_ratio_calculation(self, engine, sample_signals, sample_prices):
        """Test Sharpe ratio calculation."""
        result = engine.run(sample_signals, sample_prices)

        # Sharpe should be calculated
        assert hasattr(result, 'sharpe_ratio')
        assert isinstance(result.sharpe_ratio, float)

    def test_max_drawdown_calculation(self, engine):
        """Test max drawdown calculation."""
        # Create scenario with clear drawdown
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        # Up, then down, then up
        prices_list = list(range(100, 120)) + list(range(120, 100, -1)) + list(range(100, 110))
        prices = pd.Series(prices_list[:30], index=dates)
        signals = pd.Series([1] * 30, index=dates)  # Hold long

        result = engine.run(signals, prices)

        # Should have a drawdown
        assert result.max_drawdown > 0

    def test_equity_curve(self, engine, sample_signals, sample_prices):
        """Test equity curve generation."""
        result = engine.run(signals, sample_prices)

        assert not result.equity_curve.empty
        assert 'equity' in result.equity_curve.columns
        # Equity should start at initial capital
        # (after first return calculation)
        assert result.equity_curve['equity'].iloc[1] > 0

    def test_trade_extraction(self, engine):
        """Test trade extraction from positions."""
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        prices = pd.Series(range(100, 110), index=dates)
        # Buy, hold, sell pattern
        signals = pd.Series([1, 1, 1, 0, 0, 0, -1, -1, 0, 0], index=dates)

        result = engine.run(signals, prices)

        # Should have extracted at least one trade
        assert len(result.trades) > 0

        # Check trade structure
        trade = result.trades[0]
        assert 'entry_date' in trade
        assert 'exit_date' in trade
        assert 'return_pct' in trade
