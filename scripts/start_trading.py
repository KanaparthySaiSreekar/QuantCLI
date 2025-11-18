#!/usr/bin/env python3
"""
Start QuantCLI Trading System

This script starts the trading system in either paper trading or live trading mode.
It initializes all components, connects to brokers, and begins generating signals.

Usage:
    python scripts/start_trading.py --mode paper --capital 100000
    python scripts/start_trading.py --mode live --capital 10000

Paper trading (default): Simulates trading with no real money
Live trading: Executes real trades with real capital
"""

import sys
import argparse
import asyncio
import signal
from pathlib import Path
from datetime import datetime, time as datetime_time
from typing import Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg2
from loguru import logger

from src.core.config import ConfigManager
from src.data.orchestrator import DataOrchestrator
from src.signals.generator import SignalGenerator


class TradingSystem:
    """Main trading system coordinator."""

    def __init__(self, mode: str, capital: float, symbols: List[str]):
        """
        Initialize trading system.

        Args:
            mode: 'paper' or 'live'
            capital: Starting capital
            symbols: List of symbols to trade
        """
        self.mode = mode
        self.capital = capital
        self.initial_capital = capital
        self.symbols = symbols
        self.running = False
        self.config = ConfigManager()

        # Initialize components
        logger.info(f"Initializing {mode} trading system with ${capital:,.2f} capital")

        # Data orchestrator
        self.data_orchestrator = DataOrchestrator()
        logger.success("✓ Data orchestrator initialized")

        # Signal generator
        self.signal_generator = SignalGenerator(self.config)
        logger.success("✓ Signal generator initialized")

        # Portfolio state
        self.positions: Dict[str, int] = {}
        self.cash = capital
        self.portfolio_value = capital
        self.daily_pnl = 0.0
        self.total_signals = 0
        self.total_trades = 0

        # Database connection
        self._init_database_connection()

        logger.success(f"✅ Trading system initialized in {mode.upper()} mode")

    def _init_database_connection(self):
        """Initialize database connection for logging trades."""
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

    def _is_market_open(self) -> bool:
        """Check if US market is open (9:30 AM - 4:00 PM ET)."""
        now = datetime.now()
        market_open = datetime_time(9, 30)
        market_close = datetime_time(16, 0)

        # Check if weekend
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False

        # Check if market hours (simplified - doesn't account for holidays)
        current_time = now.time()
        return market_open <= current_time <= market_close

    async def _generate_signals(self):
        """Generate trading signals for all symbols."""
        signals = []

        for symbol in self.symbols:
            try:
                # Generate signal
                signal = self.signal_generator.generate_signal(
                    symbol=symbol,
                    portfolio_value=self.portfolio_value,
                    current_positions=self.positions
                )

                if signal and signal['action'] != 'hold':
                    signals.append(signal)
                    self.total_signals += 1

                    logger.info(
                        f"Signal: {signal['symbol']} {signal['action'].upper()} "
                        f"{signal['quantity']} @ ${signal.get('current_price', 0):.2f} "
                        f"(confidence: {signal['confidence']:.2f}, "
                        f"expected_return: {signal['expected_return']:.4f})"
                    )

                    # Log to database
                    self._log_signal(signal)

            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")

        return signals

    def _log_signal(self, signal: Dict):
        """Log signal to database."""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("""
                INSERT INTO signals (time, symbol, signal_type, quantity,
                                   expected_return, confidence, strategy, regime)
                VALUES (NOW(), %s, %s, %s, %s, %s, %s, %s)
            """, (
                signal['symbol'],
                signal['action'],
                signal['quantity'],
                signal['expected_return'],
                signal['confidence'],
                signal.get('strategy', 'ensemble'),
                signal.get('regime', 'unknown')
            ))
            self.db_conn.commit()
            cursor.close()
        except Exception as e:
            logger.error(f"Failed to log signal: {e}")

    async def _execute_signal(self, signal: Dict):
        """Execute trading signal (paper or live)."""
        symbol = signal['symbol']
        action = signal['action']
        quantity = signal['quantity']
        price = signal.get('current_price', 0)

        if self.mode == 'paper':
            # Simulate execution with realistic slippage
            slippage = 0.0005  # 0.05% slippage
            fill_price = price * (1 + slippage if action == 'buy' else 1 - slippage)
            commission = 0.005 * quantity * fill_price  # 0.5% commission

            # Check if we have enough cash to buy
            if action == 'buy':
                total_cost = quantity * fill_price + commission
                if total_cost > self.cash:
                    logger.warning(f"Insufficient cash for {symbol} buy: need ${total_cost:.2f}, have ${self.cash:.2f}")
                    return False

                # Execute buy
                self.cash -= total_cost
                self.positions[symbol] = self.positions.get(symbol, 0) + quantity

                logger.success(
                    f"✓ PAPER BUY: {quantity} {symbol} @ ${fill_price:.2f} "
                    f"(slippage: {slippage*100:.2f}%, commission: ${commission:.2f})"
                )

            elif action == 'sell':
                # Check if we have enough shares to sell
                current_position = self.positions.get(symbol, 0)
                if current_position < quantity:
                    logger.warning(f"Insufficient shares for {symbol} sell: need {quantity}, have {current_position}")
                    return False

                # Execute sell
                total_proceeds = quantity * fill_price - commission
                self.cash += total_proceeds
                self.positions[symbol] = current_position - quantity

                if self.positions[symbol] == 0:
                    del self.positions[symbol]

                logger.success(
                    f"✓ PAPER SELL: {quantity} {symbol} @ ${fill_price:.2f} "
                    f"(slippage: {slippage*100:.2f}%, commission: ${commission:.2f})"
                )

            # Log execution to database
            self._log_execution(signal, fill_price, commission, slippage)
            self.total_trades += 1
            return True

        elif self.mode == 'live':
            # TODO: Implement live trading via IBKR
            logger.warning("Live trading not yet implemented - use paper trading mode")
            return False

    def _log_execution(self, signal: Dict, fill_price: float, commission: float, slippage: float):
        """Log execution to database."""
        try:
            cursor = self.db_conn.cursor()

            # Log execution
            cursor.execute("""
                INSERT INTO executions (execution_id, order_id, time, symbol, side,
                                       quantity, fill_price, commission, slippage, venue)
                VALUES (gen_random_uuid()::text, gen_random_uuid()::text, NOW(), %s, %s, %s, %s, %s, %s, %s)
            """, (
                signal['symbol'],
                signal['action'],
                signal['quantity'],
                fill_price,
                commission,
                slippage,
                'PAPER' if self.mode == 'paper' else 'IBKR'
            ))

            self.db_conn.commit()
            cursor.close()
        except Exception as e:
            logger.error(f"Failed to log execution: {e}")

    def _update_portfolio_value(self):
        """Update portfolio value based on current positions."""
        positions_value = 0.0

        for symbol, quantity in self.positions.items():
            try:
                # Get current price
                data = self.data_orchestrator.get_daily_prices(symbol)
                if not data.empty:
                    current_price = data['close'].iloc[-1]
                    positions_value += quantity * current_price
            except Exception as e:
                logger.error(f"Error getting price for {symbol}: {e}")

        self.portfolio_value = self.cash + positions_value
        self.daily_pnl = self.portfolio_value - self.initial_capital

        # Log performance
        self._log_performance()

    def _log_performance(self):
        """Log portfolio performance to database."""
        try:
            cursor = self.db_conn.cursor()

            daily_return = (self.portfolio_value - self.initial_capital) / self.initial_capital

            cursor.execute("""
                INSERT INTO portfolio_performance
                (time, total_value, cash, positions_value, daily_pnl, daily_return, cumulative_return)
                VALUES (NOW(), %s, %s, %s, %s, %s, %s)
            """, (
                self.portfolio_value,
                self.cash,
                self.portfolio_value - self.cash,
                self.daily_pnl,
                daily_return,
                daily_return  # For now, cumulative = daily (will improve later)
            ))

            self.db_conn.commit()
            cursor.close()
        except Exception as e:
            logger.error(f"Failed to log performance: {e}")

    def _print_status(self):
        """Print current status."""
        logger.info("=" * 70)
        logger.info("PORTFOLIO STATUS")
        logger.info("=" * 70)
        logger.info(f"Total Value:    ${self.portfolio_value:,.2f}")
        logger.info(f"Cash:           ${self.cash:,.2f}")
        logger.info(f"Positions:      ${self.portfolio_value - self.cash:,.2f}")
        logger.info(f"Daily P&L:      ${self.daily_pnl:,.2f} ({(self.daily_pnl/self.initial_capital)*100:+.2f}%)")
        logger.info(f"Total Return:   {((self.portfolio_value - self.initial_capital)/self.initial_capital)*100:+.2f}%")
        logger.info("")
        logger.info(f"Signals:        {self.total_signals}")
        logger.info(f"Trades:         {self.total_trades}")
        logger.info("")

        if self.positions:
            logger.info("Current Positions:")
            for symbol, quantity in self.positions.items():
                logger.info(f"  {symbol}: {quantity} shares")
        else:
            logger.info("No open positions")

        logger.info("=" * 70)

    async def run(self):
        """Main trading loop."""
        self.running = True
        logger.info("Starting trading loop...")

        iteration = 0

        try:
            while self.running:
                iteration += 1

                # Check if market is open (for live trading)
                if self.mode == 'live' and not self._is_market_open():
                    logger.info("Market is closed, waiting...")
                    await asyncio.sleep(60)
                    continue

                logger.info(f"\n{'='*70}")
                logger.info(f"Trading Iteration {iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(f"{'='*70}")

                # Generate signals
                signals = await self._generate_signals()

                # Execute signals
                for signal in signals:
                    await self._execute_signal(signal)

                # Update portfolio value
                self._update_portfolio_value()

                # Print status
                self._print_status()

                # Wait before next iteration (5 minutes for paper trading, 1 minute for live)
                wait_time = 300 if self.mode == 'paper' else 60
                logger.info(f"Waiting {wait_time}s until next iteration...")
                await asyncio.sleep(wait_time)

        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down...")
        finally:
            self.shutdown()

    def shutdown(self):
        """Gracefully shutdown trading system."""
        logger.info("Shutting down trading system...")

        # Final status
        self._print_status()

        # Close database connection
        if hasattr(self, 'db_conn'):
            self.db_conn.close()
            logger.info("✓ Database connection closed")

        logger.success("✅ Trading system shutdown complete")


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description="Start QuantCLI Trading System")

    parser.add_argument(
        '--mode',
        type=str,
        choices=['paper', 'live'],
        default='paper',
        help='Trading mode: paper (simulation) or live (real money)'
    )

    parser.add_argument(
        '--capital',
        type=float,
        default=100000.0,
        help='Starting capital (default: $100,000)'
    )

    parser.add_argument(
        '--symbols',
        type=str,
        default='AAPL,MSFT,GOOGL,TSLA,NVDA',
        help='Comma-separated list of symbols to trade (default: AAPL,MSFT,GOOGL,TSLA,NVDA)'
    )

    args = parser.parse_args()

    # Parse symbols
    symbols = [s.strip().upper() for s in args.symbols.split(',')]

    # Warning for live mode
    if args.mode == 'live':
        logger.warning("=" * 70)
        logger.warning("⚠️  LIVE TRADING MODE - REAL MONEY AT RISK!")
        logger.warning("=" * 70)
        confirmation = input("Type 'YES' to confirm live trading: ")
        if confirmation != 'YES':
            logger.info("Live trading cancelled")
            sys.exit(0)

    # Initialize trading system
    logger.info("=" * 70)
    logger.info("QuantCLI Trading System")
    logger.info("=" * 70)
    logger.info(f"Mode:           {args.mode.upper()}")
    logger.info(f"Capital:        ${args.capital:,.2f}")
    logger.info(f"Symbols:        {', '.join(symbols)}")
    logger.info("=" * 70)
    logger.info("")

    try:
        trading_system = TradingSystem(
            mode=args.mode,
            capital=args.capital,
            symbols=symbols
        )

        # Run trading system
        asyncio.run(trading_system.run())

    except Exception as e:
        logger.error(f"❌ Trading system failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
