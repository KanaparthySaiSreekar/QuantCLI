"""
Position tracking and management.

Tracks:
- Current positions
- P&L calculation
- Position sizing
- Risk metrics
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime

from src.core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class Position:
    """
    Represents a trading position.

    Attributes:
        symbol: Stock symbol
        quantity: Number of shares (positive=long, negative=short)
        avg_entry_price: Average entry price
        current_price: Current market price
        unrealized_pnl: Unrealized P&L
        realized_pnl: Realized P&L from closed positions
        opened_at: Position open timestamp
        updated_at: Last update timestamp
    """
    symbol: str
    quantity: int
    avg_entry_price: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    opened_at: datetime = None
    updated_at: datetime = None

    def __post_init__(self):
        """Initialize timestamps."""
        if self.opened_at is None:
            self.opened_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()

    def is_long(self) -> bool:
        """Check if position is long."""
        return self.quantity > 0

    def is_short(self) -> bool:
        """Check if position is short."""
        return self.quantity < 0

    def is_flat(self) -> bool:
        """Check if position is flat (closed)."""
        return self.quantity == 0

    def market_value(self) -> float:
        """Calculate current market value."""
        return abs(self.quantity) * self.current_price

    def cost_basis(self) -> float:
        """Calculate cost basis."""
        return abs(self.quantity) * self.avg_entry_price

    def update_price(self, current_price: float) -> None:
        """
        Update current price and recalculate P&L.

        Args:
            current_price: New market price
        """
        self.current_price = current_price
        self.unrealized_pnl = self._calculate_unrealized_pnl()
        self.updated_at = datetime.now()

    def _calculate_unrealized_pnl(self) -> float:
        """Calculate unrealized P&L."""
        if self.quantity == 0:
            return 0.0

        price_diff = self.current_price - self.avg_entry_price
        return price_diff * self.quantity

    def return_pct(self) -> float:
        """Calculate return percentage."""
        if self.avg_entry_price == 0:
            return 0.0

        if self.quantity > 0:  # Long
            return (self.current_price - self.avg_entry_price) / self.avg_entry_price * 100
        else:  # Short
            return (self.avg_entry_price - self.current_price) / self.avg_entry_price * 100


class PositionManager:
    """
    Manages trading positions.

    Provides:
    - Position tracking
    - P&L calculation
    - Position updates
    - Portfolio metrics
    """

    def __init__(self):
        """Initialize position manager."""
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []

        self.logger = logger

    def open_position(
        self,
        symbol: str,
        quantity: int,
        entry_price: float
    ) -> Position:
        """
        Open new position.

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            entry_price: Entry price

        Returns:
            Position object
        """
        if symbol in self.positions and not self.positions[symbol].is_flat():
            raise ValueError(f"Position already exists for {symbol}")

        position = Position(
            symbol=symbol,
            quantity=quantity,
            avg_entry_price=entry_price,
            current_price=entry_price
        )

        self.positions[symbol] = position

        self.logger.info(
            f"Opened position: {quantity} {symbol} @ {entry_price:.2f}"
        )

        return position

    def update_position(
        self,
        symbol: str,
        quantity_change: int,
        price: float
    ) -> Position:
        """
        Update existing position (add/reduce shares).

        Args:
            symbol: Stock symbol
            quantity_change: Change in quantity (positive=add, negative=reduce)
            price: Transaction price

        Returns:
            Updated position
        """
        if symbol not in self.positions:
            # Create new position
            return self.open_position(symbol, quantity_change, price)

        position = self.positions[symbol]

        # Check if this is closing/reversing
        new_quantity = position.quantity + quantity_change

        if position.quantity * new_quantity < 0:
            # Position reversal (long->short or short->long)
            self.logger.warning(
                f"Position reversal for {symbol}: {position.quantity} -> {new_quantity}"
            )

        # Update avg entry price (for additions only)
        if (position.quantity > 0 and quantity_change > 0) or \
           (position.quantity < 0 and quantity_change < 0):
            # Adding to position
            total_cost = (position.avg_entry_price * abs(position.quantity) +
                         price * abs(quantity_change))
            position.avg_entry_price = total_cost / abs(new_quantity)

        elif abs(new_quantity) < abs(position.quantity):
            # Reducing position - realize P&L
            reduced_qty = abs(quantity_change)
            if position.is_long():
                realized_pnl = (price - position.avg_entry_price) * reduced_qty
            else:
                realized_pnl = (position.avg_entry_price - price) * reduced_qty

            position.realized_pnl += realized_pnl

            self.logger.info(
                f"Realized P&L for {symbol}: ${realized_pnl:.2f}"
            )

        # Update quantity
        position.quantity = new_quantity

        # If position closed, move to history
        if position.is_flat():
            self.closed_positions.append(position)
            del self.positions[symbol]

            self.logger.info(f"Closed position: {symbol}")

        position.updated_at = datetime.now()

        return position

    def update_prices(self, prices: Dict[str, float]) -> None:
        """
        Update market prices for all positions.

        Args:
            prices: Dictionary mapping symbol -> current price
        """
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].update_price(price)

    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position for symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Position or None
        """
        return self.positions.get(symbol)

    def get_all_positions(self) -> List[Position]:
        """Get all current positions."""
        return list(self.positions.values())

    def get_total_pnl(self) -> Dict[str, float]:
        """
        Calculate total P&L.

        Returns:
            Dictionary with P&L metrics
        """
        total_unrealized = sum(p.unrealized_pnl for p in self.positions.values())

        total_realized = sum(p.realized_pnl for p in self.closed_positions)
        total_realized += sum(p.realized_pnl for p in self.positions.values())

        return {
            'unrealized_pnl': total_unrealized,
            'realized_pnl': total_realized,
            'total_pnl': total_unrealized + total_realized
        }

    def get_portfolio_value(self) -> float:
        """Calculate total portfolio market value."""
        return sum(p.market_value() for p in self.positions.values())

    def get_exposure(self) -> Dict[str, float]:
        """
        Calculate portfolio exposure.

        Returns:
            Dictionary with exposure metrics
        """
        long_value = sum(
            p.market_value() for p in self.positions.values() if p.is_long()
        )

        short_value = sum(
            p.market_value() for p in self.positions.values() if p.is_short()
        )

        return {
            'long_exposure': long_value,
            'short_exposure': short_value,
            'net_exposure': long_value - short_value,
            'gross_exposure': long_value + short_value
        }
