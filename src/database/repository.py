"""
Data repositories for database access.

Provides clean interfaces for:
- Market data storage/retrieval
- Trade recording
- Signal storage
- Position tracking
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta

from .connection import DatabaseConnection
from src.core.logging_config import get_logger

logger = get_logger(__name__)


class MarketDataRepository:
    """Repository for market data operations."""

    def __init__(self, db: DatabaseConnection):
        """
        Initialize repository.

        Args:
            db: Database connection
        """
        self.db = db
        self.logger = logger

    def save_daily_data(
        self,
        symbol: str,
        data: pd.DataFrame,
        data_source: str = "unknown"
    ) -> int:
        """
        Save daily OHLCV data.

        Args:
            symbol: Stock symbol
            data: DataFrame with OHLCV columns
            data_source: Data provider name

        Returns:
            Number of rows inserted
        """
        if data.empty:
            return 0

        # Prepare data
        records = []
        for timestamp, row in data.iterrows():
            records.append((
                symbol,
                timestamp,
                float(row['open']),
                float(row['high']),
                float(row['low']),
                float(row['close']),
                int(row['volume']),
                float(row.get('adjusted_close', row['close'])),
                data_source
            ))

        # Batch insert with ON CONFLICT
        query = """
            INSERT INTO market_data_daily
            (symbol, timestamp, open, high, low, close, volume, adjusted_close, data_source)
            VALUES %s
            ON CONFLICT (symbol, timestamp)
            DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                adjusted_close = EXCLUDED.adjusted_close,
                data_source = EXCLUDED.data_source
        """

        count = self.db.execute_values(query, records)
        self.logger.info(f"Saved {count} daily records for {symbol}")

        return count

    def get_daily_data(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Retrieve daily OHLCV data.

        Args:
            symbol: Stock symbol
            start_date: Start date (None = all)
            end_date: End date (None = today)
            limit: Max number of rows

        Returns:
            DataFrame with OHLCV data
        """
        query = """
            SELECT timestamp, open, high, low, close, volume, adjusted_close
            FROM market_data_daily
            WHERE symbol = %s
        """

        params = [symbol]

        if start_date:
            query += " AND timestamp >= %s"
            params.append(start_date)

        if end_date:
            query += " AND timestamp <= %s"
            params.append(end_date)

        query += " ORDER BY timestamp ASC"

        if limit:
            query += f" LIMIT {limit}"

        results = self.db.execute_query(query, tuple(params))

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'adjusted_close'
        ])

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

        return df


class TradeRepository:
    """Repository for trade operations."""

    def __init__(self, db: DatabaseConnection):
        self.db = db
        self.logger = logger

    def save_trade(
        self,
        symbol: str,
        timestamp: datetime,
        side: str,
        quantity: float,
        price: float,
        commission: float = 0.0,
        slippage: float = 0.0,
        order_id: Optional[str] = None,
        strategy_name: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Save trade to database.

        Args:
            symbol: Stock symbol
            timestamp: Trade timestamp
            side: 'BUY' or 'SELL'
            quantity: Number of shares
            price: Execution price
            commission: Commission paid
            slippage: Slippage cost
            order_id: Order identifier
            strategy_name: Strategy that generated trade
            metadata: Additional metadata

        Returns:
            Trade ID
        """
        import json

        query = """
            INSERT INTO trades
            (symbol, timestamp, side, quantity, price, commission, slippage,
             order_id, strategy_name, metadata, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'FILLED')
            RETURNING id
        """

        params = (
            symbol, timestamp, side, quantity, price, commission, slippage,
            order_id, strategy_name, json.dumps(metadata) if metadata else None
        )

        with self.db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                trade_id = cur.fetchone()[0]

        self.logger.info(f"Saved trade {trade_id}: {side} {quantity} {symbol} @ {price}")

        return trade_id

    def get_trades(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """Retrieve trades."""
        query = "SELECT * FROM trades WHERE 1=1"
        params = []

        if symbol:
            query += " AND symbol = %s"
            params.append(symbol)

        if start_date:
            query += " AND timestamp >= %s"
            params.append(start_date)

        if end_date:
            query += " AND timestamp <= %s"
            params.append(end_date)

        query += f" ORDER BY timestamp DESC LIMIT {limit}"

        results = self.db.execute_query(query, tuple(params) if params else None)

        if not results:
            return pd.DataFrame()

        columns = [
            'id', 'symbol', 'timestamp', 'side', 'quantity', 'price',
            'commission', 'slippage', 'order_id', 'strategy_name',
            'status', 'metadata', 'created_at'
        ]

        return pd.DataFrame(results, columns=columns)


class SignalRepository:
    """Repository for trading signals."""

    def __init__(self, db: DatabaseConnection):
        self.db = db
        self.logger = logger

    def save_signal(
        self,
        symbol: str,
        timestamp: datetime,
        signal_type: str,
        strength: float,
        confidence: float,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> int:
        """Save trading signal."""
        import json

        query = """
            INSERT INTO signals
            (symbol, timestamp, signal_type, strength, confidence,
             model_name, model_version, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """

        params = (
            symbol, timestamp, signal_type, strength, confidence,
            model_name, model_version, json.dumps(metadata) if metadata else None
        )

        with self.db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                signal_id = cur.fetchone()[0]

        return signal_id

    def get_recent_signals(
        self,
        symbol: Optional[str] = None,
        hours: int = 24,
        limit: int = 100
    ) -> pd.DataFrame:
        """Get recent signals."""
        query = """
            SELECT * FROM signals
            WHERE timestamp > NOW() - INTERVAL '%s hours'
        """

        params = [hours]

        if symbol:
            query += " AND symbol = %s"
            params.append(symbol)

        query += f" ORDER BY timestamp DESC LIMIT {limit}"

        results = self.db.execute_query(query, tuple(params))

        if not results:
            return pd.DataFrame()

        columns = [
            'id', 'symbol', 'timestamp', 'signal_type', 'strength',
            'confidence', 'model_name', 'model_version', 'metadata', 'created_at'
        ]

        return pd.DataFrame(results, columns=columns)
