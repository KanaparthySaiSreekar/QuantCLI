#!/usr/bin/env python3
"""
Initialize QuantCLI Database

This script initializes the TimescaleDB database with all required tables,
hypertables, indexes, and policies. Run this once after starting docker-compose.

Usage:
    python scripts/init_database.py
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from loguru import logger

from src.core.config import ConfigManager


def wait_for_database(max_retries: int = 30, retry_delay: int = 2):
    """Wait for database to be ready."""
    config = ConfigManager()
    db_config = config.get("database", {})

    host = db_config.get("host", "localhost")
    port = db_config.get("port", 5432)
    user = db_config.get("user", "quantcli")
    password = db_config.get("password", "changeme")
    database = db_config.get("database", "quantcli")

    logger.info(f"Waiting for database at {host}:{port}...")

    for attempt in range(max_retries):
        try:
            conn = psycopg2.connect(
                host=host,
                port=port,
                user=user,
                password=password,
                database=database
            )
            conn.close()
            logger.success("Database is ready!")
            return True
        except psycopg2.OperationalError as e:
            if attempt < max_retries - 1:
                logger.warning(f"Database not ready (attempt {attempt + 1}/{max_retries}), retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                logger.error(f"Database failed to become ready after {max_retries} attempts")
                raise

    return False


def initialize_database():
    """Initialize database schema and TimescaleDB extensions."""
    config = ConfigManager()
    db_config = config.get("database", {})

    host = db_config.get("host", "localhost")
    port = db_config.get("port", 5432)
    user = db_config.get("user", "quantcli")
    password = db_config.get("password", "changeme")
    database = db_config.get("database", "quantcli")

    logger.info("Initializing database schema...")

    try:
        # Connect to database
        conn = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()

        # Check if TimescaleDB is installed
        cursor.execute("SELECT COUNT(*) FROM pg_extension WHERE extname = 'timescaledb';")
        if cursor.fetchone()[0] == 0:
            logger.info("Installing TimescaleDB extension...")
            cursor.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
            logger.success("TimescaleDB extension installed")
        else:
            logger.info("TimescaleDB extension already installed")

        # Read and execute init.sql
        init_sql_path = Path(__file__).parent.parent / "docker" / "timescaledb" / "init.sql"

        if init_sql_path.exists():
            logger.info(f"Executing initialization SQL from {init_sql_path}...")
            with open(init_sql_path, 'r') as f:
                sql_content = f.read()

            # Execute SQL (split by semicolons to handle multiple statements)
            cursor.execute(sql_content)

            logger.success("Database schema initialized successfully")
        else:
            logger.warning(f"Init SQL file not found at {init_sql_path}")

        # Verify tables were created
        cursor.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_type = 'BASE TABLE'
            ORDER BY table_name;
        """)

        tables = cursor.fetchall()
        logger.info(f"Created {len(tables)} tables:")
        for table in tables:
            logger.info(f"  ✓ {table[0]}")

        # Verify hypertables were created
        cursor.execute("SELECT hypertable_name FROM timescaledb_information.hypertables;")
        hypertables = cursor.fetchall()
        logger.info(f"Created {len(hypertables)} hypertables:")
        for hypertable in hypertables:
            logger.info(f"  ✓ {hypertable[0]} (time-series optimized)")

        # Close connection
        cursor.close()
        conn.close()

        logger.success("✅ Database initialization completed successfully!")
        return True

    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


def verify_database():
    """Verify database is properly configured."""
    config = ConfigManager()
    db_config = config.get("database", {})

    host = db_config.get("host", "localhost")
    port = db_config.get("port", 5432)
    user = db_config.get("user", "quantcli")
    password = db_config.get("password", "changeme")
    database = db_config.get("database", "quantcli")

    logger.info("Verifying database configuration...")

    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database
        )
        cursor = conn.cursor()

        # Check critical tables exist
        critical_tables = [
            'market_data', 'signals', 'orders', 'executions',
            'positions', 'portfolio_performance', 'system_events'
        ]

        for table in critical_tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table};")
            count = cursor.fetchone()[0]
            logger.info(f"  ✓ Table '{table}' exists ({count} rows)")

        # Check TimescaleDB functions
        cursor.execute("SELECT * FROM trading_system_stats;")
        stats = cursor.fetchone()
        if stats:
            logger.info("Database statistics:")
            logger.info(f"  Market data rows: {stats[0]}")
            logger.info(f"  Symbols tracked: {stats[1]}")
            logger.info(f"  Total signals: {stats[2]}")
            logger.info(f"  Total orders: {stats[3]}")
            logger.info(f"  Total executions: {stats[4]}")

        cursor.close()
        conn.close()

        logger.success("✅ Database verification passed!")
        return True

    except Exception as e:
        logger.error(f"Database verification failed: {e}")
        return False


def main():
    """Main execution."""
    logger.info("=" * 70)
    logger.info("QuantCLI Database Initialization")
    logger.info("=" * 70)

    try:
        # Wait for database to be ready
        wait_for_database()

        # Initialize database
        initialize_database()

        # Verify database
        verify_database()

        logger.info("=" * 70)
        logger.success("🎉 Database is ready for trading!")
        logger.info("=" * 70)
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Configure API keys in .env file")
        logger.info("  2. Run: python scripts/update_data.py (download market data)")
        logger.info("  3. Run: python scripts/train_ensemble.py (train models)")
        logger.info("  4. Run: python scripts/start_trading.py --mode paper")
        logger.info("")

    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
