"""
Order management system.

Handles:
- Order lifecycle
- Order validation
- Order tracking
- Fill management
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List
from datetime import datetime
from enum import Enum

from src.core.logging_config import get_logger
from src.core.exceptions import ExecutionError, ValidationError

logger = get_logger(__name__)


class OrderStatus(Enum):
    """Order status states."""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class Order:
    """
    Represents a trading order.

    Attributes:
        order_id: Unique order identifier
        symbol: Stock symbol
        action: 'BUY' or 'SELL'
        quantity: Number of shares
        order_type: Order type (MKT, LMT, etc.)
        limit_price: Limit price (for limit orders)
        stop_price: Stop price (for stop orders)
        status: Current order status
        filled_quantity: Number of shares filled
        avg_fill_price: Average fill price
        submitted_at: Submission timestamp
        filled_at: Fill timestamp
        metadata: Additional metadata
    """
    order_id: int
    symbol: str
    action: str
    quantity: int
    order_type: str = "MKT"
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    metadata: Dict = field(default_factory=dict)

    def is_buy(self) -> bool:
        """Check if buy order."""
        return self.action.upper() == 'BUY'

    def is_sell(self) -> bool:
        """Check if sell order."""
        return self.action.upper() == 'SELL'

    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.status == OrderStatus.FILLED

    def is_active(self) -> bool:
        """Check if order is active (can be filled)."""
        return self.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]

    def remaining_quantity(self) -> int:
        """Get remaining quantity to be filled."""
        return self.quantity - self.filled_quantity


class OrderManager:
    """
    Manages order lifecycle and validation.

    Provides:
    - Order creation and validation
    - Order tracking
    - Fill management
    - Order history
    """

    def __init__(self):
        """Initialize order manager."""
        self.next_order_id = 1
        self.orders: Dict[int, Order] = {}
        self.active_orders: Dict[int, Order] = {}

        self.logger = logger

    def create_order(
        self,
        symbol: str,
        action: str,
        quantity: int,
        order_type: str = "MKT",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        metadata: Optional[Dict] = None
    ) -> Order:
        """
        Create new order.

        Args:
            symbol: Stock symbol
            action: 'BUY' or 'SELL'
            quantity: Number of shares
            order_type: Order type
            limit_price: Limit price
            stop_price: Stop price
            metadata: Additional metadata

        Returns:
            Order object

        Raises:
            ValidationError: If order parameters invalid
        """
        # Validate
        self._validate_order(symbol, action, quantity, order_type, limit_price, stop_price)

        # Create order
        order_id = self.next_order_id
        self.next_order_id += 1

        order = Order(
            order_id=order_id,
            symbol=symbol,
            action=action.upper(),
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            status=OrderStatus.PENDING,
            metadata=metadata or {}
        )

        self.orders[order_id] = order
        self.active_orders[order_id] = order

        self.logger.info(
            f"Created order {order_id}: {action} {quantity} {symbol} {order_type}"
        )

        return order

    def _validate_order(
        self,
        symbol: str,
        action: str,
        quantity: int,
        order_type: str,
        limit_price: Optional[float],
        stop_price: Optional[float]
    ) -> None:
        """Validate order parameters."""
        if not symbol or not isinstance(symbol, str):
            raise ValidationError("Symbol must be non-empty string")

        if action.upper() not in ['BUY', 'SELL']:
            raise ValidationError(f"Invalid action: {action}. Must be BUY or SELL")

        if quantity <= 0:
            raise ValidationError(f"Quantity must be positive: {quantity}")

        if order_type not in ['MKT', 'LMT', 'STP', 'STP LMT']:
            raise ValidationError(f"Invalid order type: {order_type}")

        if order_type == 'LMT' and limit_price is None:
            raise ValidationError("Limit price required for limit orders")

        if order_type in ['STP', 'STP LMT'] and stop_price is None:
            raise ValidationError("Stop price required for stop orders")

        if limit_price is not None and limit_price <= 0:
            raise ValidationError(f"Limit price must be positive: {limit_price}")

        if stop_price is not None and stop_price <= 0:
            raise ValidationError(f"Stop price must be positive: {stop_price}")

    def submit_order(self, order_id: int) -> None:
        """
        Mark order as submitted.

        Args:
            order_id: Order to submit
        """
        if order_id not in self.orders:
            raise ExecutionError(f"Order {order_id} not found")

        order = self.orders[order_id]

        if order.status != OrderStatus.PENDING:
            raise ExecutionError(
                f"Cannot submit order {order_id}: status={order.status.value}"
            )

        order.status = OrderStatus.SUBMITTED
        order.submitted_at = datetime.now()

        self.logger.info(f"Submitted order {order_id}")

    def fill_order(
        self,
        order_id: int,
        filled_quantity: int,
        fill_price: float
    ) -> None:
        """
        Record order fill.

        Args:
            order_id: Order that was filled
            filled_quantity: Quantity filled
            fill_price: Fill price

        Raises:
            ExecutionError: If order cannot be filled
        """
        if order_id not in self.orders:
            raise ExecutionError(f"Order {order_id} not found")

        order = self.orders[order_id]

        if not order.is_active():
            raise ExecutionError(
                f"Cannot fill order {order_id}: status={order.status.value}"
            )

        # Update fill information
        total_filled = order.filled_quantity + filled_quantity

        if total_filled > order.quantity:
            raise ExecutionError(
                f"Fill quantity {filled_quantity} exceeds remaining {order.remaining_quantity()}"
            )

        # Calculate average fill price
        total_value = (order.avg_fill_price * order.filled_quantity +
                      fill_price * filled_quantity)
        order.avg_fill_price = total_value / total_filled
        order.filled_quantity = total_filled

        # Update status
        if total_filled == order.quantity:
            order.status = OrderStatus.FILLED
            order.filled_at = datetime.now()

            # Remove from active orders
            if order_id in self.active_orders:
                del self.active_orders[order_id]

            self.logger.info(
                f"Order {order_id} filled: {order.filled_quantity} @ {order.avg_fill_price:.2f}"
            )
        else:
            order.status = OrderStatus.PARTIALLY_FILLED

            self.logger.info(
                f"Order {order_id} partially filled: {filled_quantity} @ {fill_price:.2f} "
                f"({order.filled_quantity}/{order.quantity} total)"
            )

    def cancel_order(self, order_id: int) -> None:
        """
        Cancel order.

        Args:
            order_id: Order to cancel
        """
        if order_id not in self.orders:
            raise ExecutionError(f"Order {order_id} not found")

        order = self.orders[order_id]

        if not order.is_active():
            raise ExecutionError(
                f"Cannot cancel order {order_id}: status={order.status.value}"
            )

        order.status = OrderStatus.CANCELLED

        # Remove from active orders
        if order_id in self.active_orders:
            del self.active_orders[order_id]

        self.logger.info(f"Cancelled order {order_id}")

    def get_order(self, order_id: int) -> Optional[Order]:
        """
        Get order by ID.

        Args:
            order_id: Order ID

        Returns:
            Order object or None
        """
        return self.orders.get(order_id)

    def get_active_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """
        Get active orders.

        Args:
            symbol: Filter by symbol (None = all)

        Returns:
            List of active orders
        """
        orders = list(self.active_orders.values())

        if symbol:
            orders = [o for o in orders if o.symbol == symbol]

        return orders

    def get_order_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> List[Order]:
        """
        Get order history.

        Args:
            symbol: Filter by symbol (None = all)
            limit: Maximum number of orders

        Returns:
            List of orders
        """
        orders = list(self.orders.values())

        if symbol:
            orders = [o for o in orders if o.symbol == symbol]

        # Sort by submission time (most recent first)
        orders.sort(
            key=lambda x: x.submitted_at or datetime.min,
            reverse=True
        )

        return orders[:limit]

    def get_fills(self, symbol: Optional[str] = None) -> List[Order]:
        """
        Get filled orders.

        Args:
            symbol: Filter by symbol

        Returns:
            List of filled orders
        """
        fills = [
            o for o in self.orders.values()
            if o.status == OrderStatus.FILLED
        ]

        if symbol:
            fills = [o for o in fills if o.symbol == symbol]

        return fills
