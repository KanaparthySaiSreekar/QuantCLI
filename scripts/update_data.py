#!/usr/bin/env python3
"""
Update Market Data

This script downloads and updates market data for all tracked symbols.
Run this daily to keep your data up-to-date.

Usage:
    python scripts/update_data.py
    python scripts/update_data.py --symbols AAPL,MSFT,GOOGL
    python scripts/update_data.py --days 365
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg2
import pandas as pd
from loguru import logger

from src.core.config import ConfigManager
from src.data.orchestrator import DataOrchestrator


class DataUpdater:
    """Market data updater."""

    def __init__(self):
        """Initialize data updater."""
        self.config = ConfigManager()
        self.orchestrator = DataOrchestrator()
        self._init_database_connection()

    def _init_database_connection(self):
        """Initialize database connection."""
        db_config = self.config.get("database", {})

        try:
            self.db_conn = psycopg2.connect(
                host=db_config.get("host", "localhost"),
                port=db_config.get("port", 5432),
                user=db_config.get("user", "quantcli"),
                password=db_config.get("password", "changeme"),
                database=db_config.get("database", "quantcli")
            )
            logger.success("✓ Database connection established")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def update_market_data(self, symbols: List[str], days: int = 365):
        """
        Update market data for symbols.

        Args:
            symbols: List of symbols to update
            days: Number of days of historical data to fetch
        """
        logger.info(f"Updating market data for {len(symbols)} symbols ({days} days)")

        start_date = datetime.now() - timedelta(days=days)
        success_count = 0
        error_count = 0

        for i, symbol in enumerate(symbols, 1):
            try:
                logger.info(f"[{i}/{len(symbols)}] Fetching {symbol}...")

                # Get data from orchestrator (with automatic failover)
                data = self.orchestrator.get_daily_prices(
                    symbol=symbol,
                    start_date=start_date,
                    use_failover=True
                )

                if data.empty:
                    logger.warning(f"No data returned for {symbol}")
                    error_count += 1
                    continue

                # Store in database
                self._store_market_data(symbol, data)

                logger.success(f"✓ {symbol}: {len(data)} rows stored")
                success_count += 1

            except Exception as e:
                logger.error(f"✗ {symbol}: {e}")
                error_count += 1

        logger.info("=" * 70)
        logger.success(f"✅ Updated {success_count} symbols successfully")
        if error_count > 0:
            logger.warning(f"⚠️  {error_count} symbols failed")
        logger.info("=" * 70)

    def _store_market_data(self, symbol: str, data: pd.DataFrame):
        """Store market data in database."""
        cursor = self.db_conn.cursor()

        for idx, row in data.iterrows():
            try:
                cursor.execute("""
                    INSERT INTO market_data (time, symbol, open, high, low, close, volume, adj_close, data_source)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (time, symbol) DO UPDATE
                    SET open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume,
                        adj_close = EXCLUDED.adj_close,
                        data_source = EXCLUDED.data_source
                """, (
                    idx,
                    symbol,
                    row.get('open'),
                    row.get('high'),
                    row.get('low'),
                    row.get('close'),
                    row.get('volume'),
                    row.get('adj_close', row.get('close')),
                    row.get('source', 'unknown')
                ))
            except Exception as e:
                logger.error(f"Error storing row for {symbol} at {idx}: {e}")

        self.db_conn.commit()
        cursor.close()

    def get_tracked_symbols(self) -> List[str]:
        """Get list of symbols currently tracked in database."""
        cursor = self.db_conn.cursor()
        cursor.execute("SELECT DISTINCT symbol FROM market_data ORDER BY symbol;")
        symbols = [row[0] for row in cursor.fetchall()]
        cursor.close()
        return symbols

    def get_data_stats(self):
        """Get data statistics."""
        cursor = self.db_conn.cursor()

        # Total rows
        cursor.execute("SELECT COUNT(*) FROM market_data;")
        total_rows = cursor.fetchone()[0]

        # Unique symbols
        cursor.execute("SELECT COUNT(DISTINCT symbol) FROM market_data;")
        unique_symbols = cursor.fetchone()[0]

        # Date range
        cursor.execute("SELECT MIN(time), MAX(time) FROM market_data;")
        min_date, max_date = cursor.fetchone()

        # Latest data per symbol
        cursor.execute("""
            SELECT symbol, MAX(time) as latest_date
            FROM market_data
            GROUP BY symbol
            ORDER BY latest_date DESC
            LIMIT 10;
        """)
        latest_data = cursor.fetchall()

        cursor.close()

        logger.info("=" * 70)
        logger.info("DATA STATISTICS")
        logger.info("=" * 70)
        logger.info(f"Total rows:       {total_rows:,}")
        logger.info(f"Unique symbols:   {unique_symbols}")
        logger.info(f"Date range:       {min_date} to {max_date}")
        logger.info("")
        logger.info("Latest data by symbol:")
        for symbol, latest_date in latest_data:
            logger.info(f"  {symbol}: {latest_date}")
        logger.info("=" * 70)

    def close(self):
        """Close database connection."""
        if hasattr(self, 'db_conn'):
            self.db_conn.close()


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description="Update QuantCLI Market Data")

    parser.add_argument(
        '--symbols',
        type=str,
        default=None,
        help='Comma-separated list of symbols to update (default: all tracked symbols)'
    )

    parser.add_argument(
        '--days',
        type=int,
        default=365,
        help='Number of days of historical data to fetch (default: 365)'
    )

    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show data statistics only (no update)'
    )

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("QuantCLI Market Data Updater")
    logger.info("=" * 70)
    logger.info("")

    try:
        updater = DataUpdater()

        # Show stats if requested
        if args.stats:
            updater.get_data_stats()
            updater.close()
            return

        # Get symbols to update
        if args.symbols:
            symbols = [s.strip().upper() for s in args.symbols.split(',')]
            logger.info(f"Updating {len(symbols)} specified symbols")
        else:
            # Use default symbols or tracked symbols
            tracked = updater.get_tracked_symbols()
            if tracked:
                symbols = tracked
                logger.info(f"Updating {len(symbols)} tracked symbols")
            else:
                # Default symbols for initial setup
                symbols = [
                    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
                    'NVDA', 'META', 'JPM', 'V', 'WMT',
                    'SPY', 'QQQ', 'IWM'  # ETFs for market indicators
                ]
                logger.info(f"No tracked symbols found, using default {len(symbols)} symbols")

        logger.info(f"Fetching {args.days} days of historical data")
        logger.info("")

        # Update data
        updater.update_market_data(symbols, args.days)

        # Show stats
        updater.get_data_stats()

        # Close
        updater.close()

        logger.info("")
        logger.success("✅ Data update completed!")

    except Exception as e:
        logger.error(f"❌ Data update failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
