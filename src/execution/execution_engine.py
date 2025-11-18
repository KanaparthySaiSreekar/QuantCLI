"""
Execution engine.

Orchestrates:
- Signal to order conversion
- Order execution via broker
- Position management
- Risk checks
- Trade recording
"""

from typing import Optional, Dict, List
from datetime import datetime

from .broker import IBKRClient
from .order_manager import OrderManager, Order
from .position_manager import PositionManager
from src.signals.generator import Signal, SignalType
from src.core.logging_config import get_logger
from src.core.exceptions import ExecutionError, RiskError

logger = get_logger(__name__)


class ExecutionEngine:
    """
    Main execution engine.

    Coordinates signal execution with risk management.
    """

    def __init__(
        self,
        broker: IBKRClient,
        order_manager: Optional[OrderManager] = None,
        position_manager: Optional[PositionManager] = None,
        max_position_size: float = 0.10,  # 10% of portfolio per position
        max_portfolio_exposure: float = 1.0  # 100% gross exposure
    ):
        """
        Initialize execution engine.

        Args:
            broker: Broker client
            order_manager: Order manager instance
            position_manager: Position manager instance
            max_position_size: Max position size as fraction of portfolio
            max_portfolio_exposure: Max gross exposure
        """
        self.broker = broker
        self.order_manager = order_manager or OrderManager()
        self.position_manager = position_manager or PositionManager()

        self.max_position_size = max_position_size
        self.max_portfolio_exposure = max_portfolio_exposure

        self.portfolio_value = 100000.0  # Starting capital

        self.logger = logger

    def execute_signal(
        self,
        signal: Signal,
        current_price: float,
        portfolio_value: Optional[float] = None
    ) -> Optional[int]:
        """
        Execute trading signal.

        Args:
            signal: Trading signal
            current_price: Current market price
            portfolio_value: Current portfolio value

        Returns:
            Order ID if order placed, None otherwise
        """
        if portfolio_value:
            self.portfolio_value = portfolio_value

        # Check if signal is strong enough
        if signal.signal_type == SignalType.HOLD:
            self.logger.info(f"{signal.symbol}: HOLD signal, no action")
            return None

        # Get current position
        current_position = self.position_manager.get_position(signal.symbol)

        # Determine desired position
        desired_quantity = self._calculate_position_size(
            signal, current_price
        )

        if desired_quantity == 0:
            self.logger.info(f"{signal.symbol}: Position size is zero, skipping")
            return None

        # Calculate order quantity
        current_quantity = current_position.quantity if current_position else 0

        if signal.signal_type == SignalType.BUY:
            order_quantity = desired_quantity - current_quantity
            action = 'BUY'
        else:  # SELL
            order_quantity = -(desired_quantity + current_quantity)
            action = 'SELL'

        if order_quantity == 0:
            self.logger.info(f"{signal.symbol}: Already at desired position")
            return None

        # Make quantity positive
        order_quantity = abs(order_quantity)

        # Pre-trade risk checks
        try:
            self._pre_trade_risk_check(signal.symbol, action, order_quantity, current_price)
        except RiskError as e:
            self.logger.warning(f"Risk check failed for {signal.symbol}: {e}")
            return None

        # Create and submit order
        order = self.order_manager.create_order(
            symbol=signal.symbol,
            action=action,
            quantity=order_quantity,
            order_type='MKT',
            metadata={
                'signal_strength': signal.strength,
                'signal_confidence': signal.confidence,
                'signal_metadata': signal.metadata
            }
        )

        # Submit to broker
        try:
            broker_order_id = self.broker.place_order(
                symbol=signal.symbol,
                action=action,
                quantity=order_quantity,
                order_type='MKT'
            )

            self.order_manager.submit_order(order.order_id)

            self.logger.info(
                f"Executed: {action} {order_quantity} {signal.symbol} @ market "
                f"(broker order {broker_order_id})"
            )

            # Mock fill for market orders
            self.order_manager.fill_order(
                order.order_id,
                order_quantity,
                current_price
            )

            # Update position
            quantity_change = order_quantity if action == 'BUY' else -order_quantity
            self.position_manager.update_position(
                signal.symbol,
                quantity_change,
                current_price
            )

            return order.order_id

        except Exception as e:
            self.logger.error(f"Order execution failed: {e}")
            raise ExecutionError(f"Failed to execute order: {e}") from e

    def _calculate_position_size(
        self,
        signal: Signal,
        current_price: float
    ) -> int:
        """
        Calculate position size based on signal strength and portfolio value.

        Args:
            signal: Trading signal
            current_price: Current price

        Returns:
            Number of shares
        """
        # Position value based on signal strength
        base_position_value = self.portfolio_value * self.max_position_size

        # Adjust by signal strength and confidence
        risk_factor = signal.strength * signal.confidence
        position_value = base_position_value * risk_factor

        # Convert to shares
        shares = int(position_value / current_price)

        return shares

    def _pre_trade_risk_check(
        self,
        symbol: str,
        action: str,
        quantity: int,
        price: float
    ) -> None:
        """
        Perform pre-trade risk checks.

        Args:
            symbol: Stock symbol
            action: BUY or SELL
            quantity: Order quantity
            price: Order price

        Raises:
            RiskError: If risk check fails
        """
        # Check position size
        position_value = quantity * price
        if position_value > self.portfolio_value * self.max_position_size:
            raise RiskError(
                f"Position size ${position_value:.2f} exceeds limit "
                f"${self.portfolio_value * self.max_position_size:.2f}"
            )

        # Check portfolio exposure
        exposure = self.position_manager.get_exposure()
        new_exposure = exposure['gross_exposure'] + position_value

        if new_exposure > self.portfolio_value * self.max_portfolio_exposure:
            raise RiskError(
                f"Gross exposure ${new_exposure:.2f} exceeds limit "
                f"${self.portfolio_value * self.max_portfolio_exposure:.2f}"
            )

        self.logger.debug(f"Pre-trade risk check passed for {symbol}")

    def execute_batch_signals(
        self,
        signals: List[Signal],
        prices: Dict[str, float]
    ) -> Dict[str, Optional[int]]:
        """
        Execute multiple signals in batch.

        Args:
            signals: List of signals
            prices: Dictionary mapping symbol -> current price

        Returns:
            Dictionary mapping symbol -> order_id (or None)
        """
        results = {}

        for signal in signals:
            if signal.symbol not in prices:
                self.logger.warning(f"No price available for {signal.symbol}")
                results[signal.symbol] = None
                continue

            try:
                order_id = self.execute_signal(
                    signal,
                    prices[signal.symbol]
                )
                results[signal.symbol] = order_id

            except Exception as e:
                self.logger.error(f"Failed to execute {signal.symbol}: {e}")
                results[signal.symbol] = None

        return results

    def get_portfolio_summary(self) -> Dict:
        """
        Get portfolio summary.

        Returns:
            Dictionary with portfolio metrics
        """
        pnl = self.position_manager.get_total_pnl()
        exposure = self.position_manager.get_exposure()
        positions = self.position_manager.get_all_positions()

        return {
            'portfolio_value': self.portfolio_value,
            'n_positions': len(positions),
            'unrealized_pnl': pnl['unrealized_pnl'],
            'realized_pnl': pnl['realized_pnl'],
            'total_pnl': pnl['total_pnl'],
            'long_exposure': exposure['long_exposure'],
            'short_exposure': exposure['short_exposure'],
            'net_exposure': exposure['net_exposure'],
            'gross_exposure': exposure['gross_exposure'],
            'positions': [
                {
                    'symbol': p.symbol,
                    'quantity': p.quantity,
                    'avg_price': p.avg_entry_price,
                    'current_price': p.current_price,
                    'unrealized_pnl': p.unrealized_pnl,
                    'return_pct': p.return_pct()
                }
                for p in positions
            ]
        }
