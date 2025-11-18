"""
Interactive Brokers client integration.

Provides interface to IBKR TWS API for:
- Order placement
- Position retrieval
- Account information
- Market data streaming
"""

from typing import Optional, Dict, List, Callable, Any
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from src.core.logging_config import get_logger
from src.core.exceptions import ExecutionError

logger = get_logger(__name__)


class OrderType(Enum):
    """Order types."""
    MARKET = "MKT"
    LIMIT = "LMT"
    STOP = "STP"
    STOP_LIMIT = "STP LMT"


@dataclass
class IBKRConfig:
    """IBKR configuration."""
    host: str = "127.0.0.1"
    port: int = 7497  # TWS paper trading port
    client_id: int = 1
    account: str = ""


class IBKRClient:
    """
    Interactive Brokers API client.

    Note: This is a mock implementation. For production, use the official
    IBKR TWS API with the `ib_insync` library.

    Example:
        >>> client = IBKRClient(config)
        >>> client.connect()
        >>> order_id = client.place_order('AAPL', 'BUY', 100, 'MKT')
        >>> client.disconnect()
    """

    def __init__(self, config: Optional[IBKRConfig] = None):
        """
        Initialize IBKR client.

        Args:
            config: IBKR configuration
        """
        self.config = config or IBKRConfig()
        self.is_connected = False
        self.next_order_id = 1
        self.positions: Dict[str, int] = {}
        self.orders: Dict[int, Dict] = {}

        self.logger = logger

    def connect(self) -> None:
        """
        Connect to IBKR TWS.

        In production, this would establish connection to TWS API.
        """
        self.logger.info(
            f"Connecting to IBKR TWS at {self.config.host}:{self.config.port}"
        )

        # Mock connection
        self.is_connected = True

        self.logger.info("Connected to IBKR TWS (mock mode)")

    def disconnect(self) -> None:
        """Disconnect from IBKR TWS."""
        if self.is_connected:
            self.is_connected = False
            self.logger.info("Disconnected from IBKR TWS")

    def place_order(
        self,
        symbol: str,
        action: str,  # BUY or SELL
        quantity: int,
        order_type: str = "MKT",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> int:
        """
        Place order.

        Args:
            symbol: Stock symbol
            action: 'BUY' or 'SELL'
            quantity: Number of shares
            order_type: Order type (MKT, LMT, STP, STP LMT)
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders

        Returns:
            Order ID

        Raises:
            ExecutionError: If order placement fails
        """
        if not self.is_connected:
            raise ExecutionError("Not connected to IBKR")

        order_id = self.next_order_id
        self.next_order_id += 1

        order = {
            'order_id': order_id,
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'order_type': order_type,
            'limit_price': limit_price,
            'stop_price': stop_price,
            'status': 'SUBMITTED',
            'filled_quantity': 0,
            'avg_fill_price': 0.0,
            'submitted_at': datetime.now()
        }

        self.orders[order_id] = order

        self.logger.info(
            f"Placed order {order_id}: {action} {quantity} {symbol} {order_type}"
        )

        # Mock: immediately fill market orders
        if order_type == "MKT":
            self._mock_fill_order(order_id)

        return order_id

    def _mock_fill_order(self, order_id: int, fill_price: float = 100.0) -> None:
        """Mock order fill (for testing)."""
        if order_id not in self.orders:
            return

        order = self.orders[order_id]
        order['status'] = 'FILLED'
        order['filled_quantity'] = order['quantity']
        order['avg_fill_price'] = fill_price
        order['filled_at'] = datetime.now()

        # Update positions
        symbol = order['symbol']
        quantity = order['quantity'] if order['action'] == 'BUY' else -order['quantity']

        self.positions[symbol] = self.positions.get(symbol, 0) + quantity

        self.logger.info(
            f"Order {order_id} filled: {order['quantity']} @ {fill_price}"
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

        if order['status'] in ['FILLED', 'CANCELLED']:
            raise ExecutionError(
                f"Cannot cancel order {order_id}: status={order['status']}"
            )

        order['status'] = 'CANCELLED'
        order['cancelled_at'] = datetime.now()

        self.logger.info(f"Cancelled order {order_id}")

    def get_order_status(self, order_id: int) -> Optional[Dict]:
        """
        Get order status.

        Args:
            order_id: Order ID

        Returns:
            Order details dictionary
        """
        return self.orders.get(order_id)

    def get_positions(self) -> Dict[str, int]:
        """
        Get current positions.

        Returns:
            Dictionary mapping symbol -> quantity
        """
        return self.positions.copy()

    def get_account_summary(self) -> Dict[str, float]:
        """
        Get account summary.

        Returns:
            Dictionary with account metrics
        """
        return {
            'net_liquidation': 100000.0,
            'total_cash_value': 50000.0,
            'gross_position_value': 50000.0,
            'buying_power': 200000.0
        }

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
        return False
