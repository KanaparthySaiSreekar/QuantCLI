"""
Context Provider for QuantCLI Copilot.

Gathers relevant context from the trading system to provide
informed AI assistance.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import pandas as pd
from loguru import logger

from ..core.config import ConfigManager
from ..database.connection import DatabaseConnection


class ContextProvider:
    """
    Provides context from the trading system for AI copilot.

    Gathers information from:
    - Database (positions, trades, performance)
    - Configuration (current settings)
    - Market data (recent prices, indicators)
    - Models (predictions, performance)
    """

    def __init__(self):
        """Initialize the context provider."""
        self.config = ConfigManager()
        self.db = None

    def _get_db(self) -> DatabaseConnection:
        """Get database connection (lazy initialization)."""
        if self.db is None:
            self.db = DatabaseConnection()
        return self.db

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Get current portfolio summary.

        Returns:
            Portfolio data including total value, positions, cash, etc.
        """
        try:
            db = self._get_db()
            with db.get_session() as session:
                # Query current positions
                query = """
                    SELECT
                        symbol,
                        quantity,
                        avg_price,
                        current_price,
                        (current_price - avg_price) * quantity as unrealized_pnl,
                        (current_price - avg_price) / avg_price * 100 as pnl_pct
                    FROM positions
                    WHERE quantity != 0
                    ORDER BY abs((current_price - avg_price) * quantity) DESC
                """

                positions = pd.read_sql(query, session.connection())

                # Calculate summary
                total_value = positions['current_price'].multiply(positions['quantity']).sum()
                total_pnl = positions['unrealized_pnl'].sum()

                return {
                    'total_value': float(total_value) if not pd.isna(total_value) else 0.0,
                    'total_unrealized_pnl': float(total_pnl) if not pd.isna(total_pnl) else 0.0,
                    'num_positions': len(positions),
                    'largest_position': positions.iloc[0].to_dict() if len(positions) > 0 else None,
                    'positions_summary': positions.head(10).to_dict('records'),
                }

        except Exception as e:
            logger.warning(f"Could not get portfolio summary: {e}")
            return {
                'total_value': 0.0,
                'total_unrealized_pnl': 0.0,
                'num_positions': 0,
                'error': str(e),
            }

    def get_current_positions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get current open positions.

        Args:
            limit: Maximum number of positions to return

        Returns:
            List of position dictionaries
        """
        try:
            db = self._get_db()
            with db.get_session() as session:
                query = f"""
                    SELECT
                        symbol,
                        quantity,
                        avg_price,
                        current_price,
                        (current_price - avg_price) * quantity as unrealized_pnl,
                        last_updated
                    FROM positions
                    WHERE quantity != 0
                    ORDER BY abs(quantity * current_price) DESC
                    LIMIT {limit}
                """

                positions = pd.read_sql(query, session.connection())
                return positions.to_dict('records')

        except Exception as e:
            logger.warning(f"Could not get positions: {e}")
            return []

    def get_recent_trades(
        self,
        symbol: Optional[str] = None,
        days: int = 7,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get recent trades.

        Args:
            symbol: Filter by symbol (optional)
            days: Number of days to look back
            limit: Maximum trades to return

        Returns:
            List of trade dictionaries
        """
        try:
            db = self._get_db()
            with db.get_session() as session:
                where_clause = ""
                if symbol:
                    where_clause = f"WHERE symbol = '{symbol}' AND "
                else:
                    where_clause = "WHERE "

                cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

                query = f"""
                    SELECT
                        symbol,
                        side,
                        quantity,
                        price,
                        realized_pnl,
                        timestamp
                    FROM trades
                    {where_clause} timestamp >= '{cutoff_date}'
                    ORDER BY timestamp DESC
                    LIMIT {limit}
                """

                trades = pd.read_sql(query, session.connection())
                return trades.to_dict('records')

        except Exception as e:
            logger.warning(f"Could not get recent trades: {e}")
            return []

    def get_performance_metrics(self, days: int = 30) -> Dict[str, float]:
        """
        Get recent performance metrics.

        Args:
            days: Number of days to analyze

        Returns:
            Performance metrics dictionary
        """
        try:
            db = self._get_db()
            with db.get_session() as session:
                cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

                query = f"""
                    SELECT
                        SUM(realized_pnl) as total_pnl,
                        COUNT(*) as num_trades,
                        COUNT(CASE WHEN realized_pnl > 0 THEN 1 END) as winning_trades,
                        COUNT(CASE WHEN realized_pnl < 0 THEN 1 END) as losing_trades,
                        AVG(realized_pnl) as avg_pnl,
                        MAX(realized_pnl) as max_win,
                        MIN(realized_pnl) as max_loss
                    FROM trades
                    WHERE timestamp >= '{cutoff_date}'
                    AND realized_pnl IS NOT NULL
                """

                result = pd.read_sql(query, session.connection())

                if len(result) > 0:
                    row = result.iloc[0]
                    win_rate = (row['winning_trades'] / row['num_trades'] * 100
                               if row['num_trades'] > 0 else 0)

                    return {
                        'total_pnl': float(row['total_pnl']) if not pd.isna(row['total_pnl']) else 0.0,
                        'num_trades': int(row['num_trades']),
                        'win_rate': float(win_rate),
                        'avg_pnl': float(row['avg_pnl']) if not pd.isna(row['avg_pnl']) else 0.0,
                        'max_win': float(row['max_win']) if not pd.isna(row['max_win']) else 0.0,
                        'max_loss': float(row['max_loss']) if not pd.isna(row['max_loss']) else 0.0,
                    }

        except Exception as e:
            logger.warning(f"Could not get performance metrics: {e}")

        return {
            'total_pnl': 0.0,
            'num_trades': 0,
            'win_rate': 0.0,
            'avg_pnl': 0.0,
            'max_win': 0.0,
            'max_loss': 0.0,
        }

    def get_recent_signals(
        self,
        symbol: Optional[str] = None,
        hours: int = 24,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get recent trading signals.

        Args:
            symbol: Filter by symbol (optional)
            hours: Hours to look back
            limit: Maximum signals to return

        Returns:
            List of signal dictionaries
        """
        try:
            db = self._get_db()
            with db.get_session() as session:
                where_clause = ""
                if symbol:
                    where_clause = f"WHERE symbol = '{symbol}' AND "
                else:
                    where_clause = "WHERE "

                cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()

                query = f"""
                    SELECT
                        symbol,
                        signal_type,
                        strength,
                        confidence,
                        metadata,
                        timestamp
                    FROM signals
                    {where_clause} timestamp >= '{cutoff_time}'
                    ORDER BY timestamp DESC
                    LIMIT {limit}
                """

                signals = pd.read_sql(query, session.connection())
                return signals.to_dict('records')

        except Exception as e:
            logger.warning(f"Could not get recent signals: {e}")
            return []

    def get_market_data(
        self,
        symbol: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get recent market data for a symbol.

        Args:
            symbol: Stock symbol
            days: Days of data to retrieve

        Returns:
            Market data summary
        """
        try:
            db = self._get_db()
            with db.get_session() as session:
                cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

                query = f"""
                    SELECT
                        timestamp,
                        open,
                        high,
                        low,
                        close,
                        volume
                    FROM market_data
                    WHERE symbol = '{symbol}'
                    AND timestamp >= '{cutoff_date}'
                    ORDER BY timestamp DESC
                """

                data = pd.read_sql(query, session.connection())

                if len(data) > 0:
                    latest = data.iloc[0]
                    returns = data['close'].pct_change()

                    return {
                        'symbol': symbol,
                        'latest_price': float(latest['close']),
                        'latest_volume': float(latest['volume']),
                        'price_change_pct': float((data.iloc[0]['close'] - data.iloc[-1]['close']) /
                                                 data.iloc[-1]['close'] * 100) if len(data) > 1 else 0.0,
                        'volatility': float(returns.std() * 100),
                        'avg_volume': float(data['volume'].mean()),
                        'high_52w': float(data['high'].max()),
                        'low_52w': float(data['low'].min()),
                    }

        except Exception as e:
            logger.warning(f"Could not get market data for {symbol}: {e}")

        return {
            'symbol': symbol,
            'error': 'Data not available',
        }

    def get_model_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about ML models.

        Args:
            model_name: Specific model name (optional)

        Returns:
            Model information and metrics
        """
        try:
            # This would typically query MLflow or model registry
            # For now, return placeholder data
            return {
                'model_name': model_name or 'ensemble_stack',
                'status': 'active',
                'last_trained': 'recent',
                'note': 'Full MLflow integration pending',
            }

        except Exception as e:
            logger.warning(f"Could not get model info: {e}")
            return {'error': str(e)}

    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get summary of current configuration.

        Returns:
            Configuration summary
        """
        try:
            config_dict = self.config.to_dict()

            # Extract key settings
            return {
                'risk_limits': config_dict.get('risk', {}),
                'data_sources': list(config_dict.get('data_sources', {}).keys()),
                'models': list(config_dict.get('models', {}).keys()),
                'backtest_settings': config_dict.get('backtest', {}),
            }

        except Exception as e:
            logger.warning(f"Could not get config summary: {e}")
            return {'error': str(e)}

    def get_full_context(
        self,
        symbol: Optional[str] = None,
        include_trades: bool = True,
        include_signals: bool = True,
    ) -> Dict[str, Any]:
        """
        Get comprehensive context for copilot.

        Args:
            symbol: Specific symbol to focus on
            include_trades: Include recent trades
            include_signals: Include recent signals

        Returns:
            Full context dictionary
        """
        context = {
            'timestamp': datetime.now().isoformat(),
            'portfolio': self.get_portfolio_summary(),
            'positions': self.get_current_positions(limit=10),
            'performance': self.get_performance_metrics(days=30),
            'config': self.get_config_summary(),
        }

        if symbol:
            context['market_data'] = self.get_market_data(symbol, days=30)

        if include_trades:
            context['recent_trades'] = self.get_recent_trades(symbol=symbol, days=7, limit=20)

        if include_signals:
            context['recent_signals'] = self.get_recent_signals(symbol=symbol, hours=24, limit=20)

        return context
