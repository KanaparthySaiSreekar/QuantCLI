"""
Backtesting engine for trading strategies.

Provides vectorized backtesting with:
- Transaction costs
- Slippage modeling
- Position sizing
- Performance metrics (Sharpe, Sortino, max drawdown)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from src.core.logging_config import get_logger
from src.core.exceptions import ValidationError

logger = get_logger(__name__)


@dataclass
class BacktestResult:
    """
    Results from a backtest run.

    Attributes:
        total_return: Total return (%)
        annual_return: Annualized return (%)
        sharpe_ratio: Sharpe ratio
        sortino_ratio: Sortino ratio
        max_drawdown: Maximum drawdown (%)
        win_rate: Percentage of winning trades
        total_trades: Total number of trades
        avg_trade_return: Average return per trade (%)
        volatility: Annualized volatility (%)
        equity_curve: DataFrame with equity over time
        trades: List of individual trades
        metrics: Additional metrics dictionary
    """
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    avg_trade_return: float
    volatility: float
    equity_curve: pd.DataFrame
    trades: List[Dict] = field(default_factory=list)
    metrics: Dict = field(default_factory=dict)

    def __str__(self) -> str:
        """String representation of results."""
        return f"""
Backtest Results:
================
Total Return:      {self.total_return:.2f}%
Annual Return:     {self.annual_return:.2f}%
Sharpe Ratio:      {self.sharpe_ratio:.2f}
Sortino Ratio:     {self.sortino_ratio:.2f}
Max Drawdown:      {self.max_drawdown:.2f}%
Volatility:        {self.volatility:.2f}%
Win Rate:          {self.win_rate:.2f}%
Total Trades:      {self.total_trades}
Avg Trade Return:  {self.avg_trade_return:.2f}%
"""


class BacktestEngine:
    """
    Vectorized backtesting engine.

    Example:
        >>> engine = BacktestEngine(
        ...     initial_capital=100000,
        ...     commission=0.001,
        ...     slippage=0.0005
        ... )
        >>> signals = pd.Series([1, 1, 0, -1, -1, 0, 1])
        >>> prices = pd.Series([100, 101, 102, 101, 100, 99, 100])
        >>> result = engine.run(signals, prices)
        >>> print(result)
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission: float = 0.001,  # 0.1%
        slippage: float = 0.0005,   # 0.05%
        position_size: float = 1.0,   # Fraction of capital per trade
        risk_free_rate: float = 0.02  # 2% annual
    ):
        """
        Initialize backtesting engine.

        Args:
            initial_capital: Starting capital ($)
            commission: Commission rate (as fraction, e.g., 0.001 = 0.1%)
            slippage: Slippage rate (as fraction)
            position_size: Position size as fraction of capital
            risk_free_rate: Annual risk-free rate for Sharpe calculation
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.position_size = position_size
        self.risk_free_rate = risk_free_rate
        self.logger = logger

    def run(
        self,
        signals: pd.Series,
        prices: pd.Series,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> BacktestResult:
        """
        Run backtest on signals and prices.

        Args:
            signals: Trading signals (1=buy, -1=sell, 0=hold)
            prices: Price series (same index as signals)
            stop_loss: Stop loss percentage (e.g., 0.02 for 2%)
            take_profit: Take profit percentage

        Returns:
            BacktestResult object

        Raises:
            ValidationError: If inputs are invalid
        """
        self._validate_inputs(signals, prices)

        # Align signals and prices
        df = pd.DataFrame({
            'signal': signals,
            'price': prices
        }).dropna()

        if len(df) == 0:
            raise ValidationError("No valid data after alignment")

        # Calculate positions
        df['position'] = self._calculate_positions(df['signal'])

        # Apply transaction costs
        df['effective_price'] = self._apply_costs(df['price'], df['position'])

        # Calculate returns
        df['returns'] = df['effective_price'].pct_change()
        df['strategy_returns'] = df['returns'] * df['position'].shift(1)

        # Calculate equity curve
        df['equity'] = self.initial_capital * (1 + df['strategy_returns']).cumprod()

        # Extract trades
        trades = self._extract_trades(df)

        # Calculate metrics
        metrics = self._calculate_metrics(df, trades)

        result = BacktestResult(
            total_return=metrics['total_return'],
            annual_return=metrics['annual_return'],
            sharpe_ratio=metrics['sharpe_ratio'],
            sortino_ratio=metrics['sortino_ratio'],
            max_drawdown=metrics['max_drawdown'],
            win_rate=metrics['win_rate'],
            total_trades=metrics['total_trades'],
            avg_trade_return=metrics['avg_trade_return'],
            volatility=metrics['volatility'],
            equity_curve=df[['equity']].copy(),
            trades=trades,
            metrics=metrics
        )

        self.logger.info(f"Backtest complete: {metrics['total_trades']} trades, "
                        f"{metrics['total_return']:.2f}% return")

        return result

    def _validate_inputs(self, signals: pd.Series, prices: pd.Series) -> None:
        """Validate input data."""
        if signals.empty or prices.empty:
            raise ValidationError("Signals and prices cannot be empty")

        if len(signals) != len(prices):
            raise ValidationError("Signals and prices must have same length")

        if not all(s in [-1, 0, 1] for s in signals.dropna().unique()):
            raise ValidationError("Signals must be -1, 0, or 1")

    def _calculate_positions(self, signals: pd.Series) -> pd.Series:
        """
        Convert signals to positions.

        Signals (1, -1, 0) → Positions (1, -1, 0)
        Position held until signal changes.
        """
        # FIXED: Use .ffill() instead of deprecated fillna(method='ffill')
        # Forward fill signals to maintain positions
        positions = signals.replace(0, np.nan).ffill().fillna(0)
        return positions

    def _apply_costs(self, prices: pd.Series, positions: pd.Series) -> pd.Series:
        """
        Apply transaction costs (commission + slippage) to prices.

        Args:
            prices: Original prices
            positions: Position series

        Returns:
            Adjusted prices including costs
        """
        # Detect position changes (trades)
        position_changes = positions.diff().abs()

        # Total cost = commission + slippage
        total_cost = self.commission + self.slippage

        # Adjust prices for costs when position changes
        # Buy: pay higher price, Sell: receive lower price
        cost_multiplier = 1 + (position_changes * total_cost * np.sign(positions))

        effective_prices = prices * cost_multiplier

        return effective_prices

    def _extract_trades(self, df: pd.DataFrame) -> List[Dict]:
        """
        Extract individual trades from position series.

        A trade = entry → exit (position goes from 0 to non-zero to 0)
        """
        trades = []
        position = df['position'].values
        prices = df['price'].values
        dates = df.index

        in_trade = False
        entry_idx = None
        entry_price = None
        entry_position = None

        for i in range(len(position)):
            if not in_trade and position[i] != 0:
                # Enter trade
                in_trade = True
                entry_idx = i
                entry_price = prices[i]
                entry_position = position[i]

            elif in_trade and (position[i] == 0 or position[i] != entry_position):
                # Exit trade
                exit_idx = i
                exit_price = prices[i]

                # Calculate trade return
                if entry_position > 0:  # Long trade
                    trade_return = (exit_price - entry_price) / entry_price
                else:  # Short trade
                    trade_return = (entry_price - exit_price) / entry_price

                # Apply costs
                trade_return -= (self.commission + self.slippage) * 2  # Entry + exit

                trades.append({
                    'entry_date': dates[entry_idx],
                    'exit_date': dates[exit_idx],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'position': 'long' if entry_position > 0 else 'short',
                    'return_pct': trade_return * 100,
                    'holding_period': (dates[exit_idx] - dates[entry_idx]).days
                })

                # Reset for next trade
                in_trade = False
                if position[i] != 0:
                    # Immediately entering new trade
                    in_trade = True
                    entry_idx = i
                    entry_price = prices[i]
                    entry_position = position[i]

        return trades

    def _calculate_metrics(self, df: pd.DataFrame, trades: List[Dict]) -> Dict:
        """Calculate performance metrics."""
        strategy_returns = df['strategy_returns'].dropna()

        # Total return
        total_return = (df['equity'].iloc[-1] / self.initial_capital - 1) * 100

        # Annualized return
        days = (df.index[-1] - df.index[0]).days
        years = days / 365.25
        annual_return = ((1 + total_return/100) ** (1/years) - 1) * 100 if years > 0 else 0

        # Volatility
        volatility = strategy_returns.std() * np.sqrt(252) * 100  # Annualized

        # Sharpe ratio
        excess_returns = strategy_returns - (self.risk_free_rate / 252)
        sharpe_ratio = (excess_returns.mean() / excess_returns.std() * np.sqrt(252)
                       if excess_returns.std() > 0 else 0)

        # Sortino ratio (only downside deviation)
        downside_returns = strategy_returns[strategy_returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
        sortino_ratio = (excess_returns.mean() / downside_std * np.sqrt(252)
                        if downside_std > 0 else 0)

        # Maximum drawdown
        equity = df['equity']
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max * 100
        max_drawdown = abs(drawdown.min())

        # Trade statistics
        if trades:
            winning_trades = [t for t in trades if t['return_pct'] > 0]
            win_rate = len(winning_trades) / len(trades) * 100
            avg_trade_return = np.mean([t['return_pct'] for t in trades])
        else:
            win_rate = 0
            avg_trade_return = 0

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'win_rate': win_rate,
            'total_trades': len(trades),
            'avg_trade_return': avg_trade_return,
            'best_trade': max([t['return_pct'] for t in trades]) if trades else 0,
            'worst_trade': min([t['return_pct'] for t in trades]) if trades else 0,
            'avg_holding_period': np.mean([t['holding_period'] for t in trades]) if trades else 0
        }
