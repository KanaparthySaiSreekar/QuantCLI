"""
Database connection management.

Handles PostgreSQL/TimescaleDB connections with pooling.
"""

import psycopg2
from psycopg2 import pool, extras
from psycopg2.extensions import connection as pg_connection
from typing import Optional, Any, List, Dict, Tuple
from contextlib import contextmanager
from pathlib import Path

from src.core.config import DatabaseSettings
from src.core.logging_config import get_logger
from src.core.exceptions import ConfigurationError

logger = get_logger(__name__)


class DatabaseConnection:
    """
    Manages PostgreSQL/TimescaleDB connections.

    Features:
    - Connection pooling
    - Context manager support
    - Automatic retry
    - Schema initialization
    """

    def __init__(self, database_url: Optional[str] = None, pool_size: int = 20):
        """
        Initialize database connection manager.

        Args:
            database_url: PostgreSQL connection URL (None = use config)
            pool_size: Size of connection pool
        """
        if database_url:
            self.database_url = database_url
        else:
            # Load from config
            try:
                db_settings = DatabaseSettings()
                self.database_url = db_settings.database_url
                pool_size = db_settings.pool_size
            except Exception as e:
                raise ConfigurationError(
                    f"Failed to load database configuration: {e}"
                ) from e

        self.pool_size = pool_size
        self.connection_pool: Optional[pool.SimpleConnectionPool] = None

        self.logger = logger

    def initialize_pool(self) -> None:
        """
        Initialize connection pool.

        Raises:
            ConfigurationError: If initialization fails
        """
        try:
            self.connection_pool = pool.SimpleConnectionPool(
                minconn=1,
                maxconn=self.pool_size,
                dsn=self.database_url
            )

            self.logger.info(
                f"Initialized database connection pool (size={self.pool_size})"
            )

            # Test connection
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT version()")
                    version = cur.fetchone()[0]
                    self.logger.info(f"Connected to: {version}")

        except Exception as e:
            self.logger.error(f"Failed to initialize database pool: {e}")
            raise ConfigurationError(
                f"Database connection failed: {e}"
            ) from e

    @contextmanager
    def get_connection(self):
        """
        Get database connection from pool.

        Yields:
            PostgreSQL connection

        Usage:
            with db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT * FROM table")
        """
        if self.connection_pool is None:
            self.initialize_pool()

        conn = None
        try:
            conn = self.connection_pool.getconn()
            yield conn
            conn.commit()

        except Exception as e:
            if conn:
                conn.rollback()
            self.logger.error(f"Database error: {e}")
            raise

        finally:
            if conn:
                self.connection_pool.putconn(conn)

    def execute_query(
        self,
        query: str,
        params: Optional[Tuple] = None,
        fetch: bool = True
    ) -> Optional[List[Tuple]]:
        """
        Execute SQL query.

        Args:
            query: SQL query string
            params: Query parameters
            fetch: Whether to fetch results

        Returns:
            List of result tuples if fetch=True, else None
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)

                if fetch:
                    return cur.fetchall()

                return None

    def execute_many(
        self,
        query: str,
        data: List[Tuple]
    ) -> int:
        """
        Execute batch insert.

        Args:
            query: SQL query with placeholders
            data: List of parameter tuples

        Returns:
            Number of rows affected
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                extras.execute_batch(cur, query, data, page_size=1000)
                return cur.rowcount

    def execute_values(
        self,
        query: str,
        data: List[Tuple],
        template: Optional[str] = None
    ) -> int:
        """
        Execute batch insert using execute_values (faster).

        Args:
            query: SQL query with VALUES placeholder
            data: List of value tuples
            template: Optional value template

        Returns:
            Number of rows affected

        Example:
            query = "INSERT INTO table (col1, col2) VALUES %s"
            data = [(1, 'a'), (2, 'b')]
            db.execute_values(query, data)
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                extras.execute_values(
                    cur, query, data,
                    template=template,
                    page_size=1000
                )
                return cur.rowcount

    def initialize_schema(self, schema_file: Optional[Path] = None) -> None:
        """
        Initialize database schema.

        Args:
            schema_file: Path to SQL schema file (None = use default)
        """
        if schema_file is None:
            schema_file = Path(__file__).parent / "schema.sql"

        if not schema_file.exists():
            raise ConfigurationError(f"Schema file not found: {schema_file}")

        self.logger.info(f"Initializing database schema from {schema_file}")

        with open(schema_file, 'r') as f:
            schema_sql = f.read()

        # Execute schema (split by statement for better error messages)
        statements = [s.strip() for s in schema_sql.split(';') if s.strip()]

        with self.get_connection() as conn:
            with conn.cursor() as cur:
                for i, statement in enumerate(statements, 1):
                    try:
                        # Skip comments
                        if statement.startswith('--') or not statement:
                            continue

                        cur.execute(statement)
                        self.logger.debug(f"Executed statement {i}/{len(statements)}")

                    except Exception as e:
                        self.logger.error(
                            f"Failed to execute statement {i}: {statement[:100]}...\n"
                            f"Error: {e}"
                        )
                        raise

        self.logger.info("Database schema initialized successfully")

    def table_exists(self, table_name: str) -> bool:
        """
        Check if table exists.

        Args:
            table_name: Name of table

        Returns:
            True if table exists
        """
        query = """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name = %s
            )
        """

        result = self.execute_query(query, (table_name,), fetch=True)
        return result[0][0] if result else False

    def get_table_count(self, table_name: str) -> int:
        """
        Get row count for table.

        Args:
            table_name: Name of table

        Returns:
            Number of rows
        """
        query = f"SELECT COUNT(*) FROM {table_name}"
        result = self.execute_query(query, fetch=True)
        return result[0][0] if result else 0

    def vacuum_analyze(self, table_name: Optional[str] = None) -> None:
        """
        Run VACUUM ANALYZE to optimize database.

        Args:
            table_name: Specific table (None = all tables)
        """
        with self.get_connection() as conn:
            # VACUUM can't run in transaction, need to autocommit
            old_isolation = conn.isolation_level
            conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)

            try:
                with conn.cursor() as cur:
                    if table_name:
                        cur.execute(f"VACUUM ANALYZE {table_name}")
                        self.logger.info(f"Vacuumed table: {table_name}")
                    else:
                        cur.execute("VACUUM ANALYZE")
                        self.logger.info("Vacuumed all tables")

            finally:
                conn.set_isolation_level(old_isolation)

    def close(self) -> None:
        """Close all connections in pool."""
        if self.connection_pool:
            self.connection_pool.closeall()
            self.logger.info("Closed database connection pool")

    def __enter__(self):
        """Context manager entry."""
        if self.connection_pool is None:
            self.initialize_pool()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
