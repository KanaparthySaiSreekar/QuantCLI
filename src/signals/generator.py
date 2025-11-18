"""
Signal generation for trading strategies.

This module provides signal generation capabilities including:
- Single symbol signal generation
- Batch signal generation for multiple symbols
- Signal strength calculation
- Risk-adjusted position sizing
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum

from ..core.exceptions import ModelError, ValidationError


class SignalType(Enum):
    """Types of trading signals."""
    BUY = 1
    SELL = -1
    HOLD = 0


@dataclass
class Signal:
    """
    Represents a trading signal.

    Attributes:
        symbol: Ticker symbol
        timestamp: When signal was generated
        signal_type: BUY, SELL, or HOLD
        strength: Signal strength (0.0 to 1.0)
        confidence: Model confidence (0.0 to 1.0)
        metadata: Additional signal metadata (features, model scores, etc.)
    """
    symbol: str
    timestamp: datetime
    signal_type: SignalType
    strength: float
    confidence: float
    metadata: Dict[str, Any]

    def __post_init__(self):
        """Validate signal attributes."""
        if not 0.0 <= self.strength <= 1.0:
            raise ValidationError(f"Signal strength must be 0-1, got {self.strength}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValidationError(f"Confidence must be 0-1, got {self.confidence}")


class SignalGenerator:
    """
    Generates trading signals for a single symbol.

    This class integrates feature engineering, model predictions, and risk
    management to generate actionable trading signals.

    Example:
        >>> generator = SignalGenerator(symbol="AAPL")
        >>> signal = generator.generate(market_data, features)
        >>> if signal.signal_type == SignalType.BUY:
        ...     print(f"Buy signal with strength {signal.strength}")
    """

    def __init__(
        self,
        symbol: str,
        min_confidence: float = 0.6,
        min_strength: float = 0.5,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize signal generator.

        Args:
            symbol: Ticker symbol to generate signals for
            min_confidence: Minimum model confidence to generate signal (0-1)
            min_strength: Minimum signal strength threshold (0-1)
            logger: Optional logger instance

        Raises:
            ValidationError: If parameters are invalid
        """
        self.symbol = symbol
        self.min_confidence = min_confidence
        self.min_strength = min_strength
        self.logger = logger or logging.getLogger(__name__)

        if not 0.0 <= min_confidence <= 1.0:
            raise ValidationError(f"min_confidence must be 0-1, got {min_confidence}")
        if not 0.0 <= min_strength <= 1.0:
            raise ValidationError(f"min_strength must be 0-1, got {min_strength}")

    def generate(
        self,
        market_data: pd.DataFrame,
        features: Optional[pd.DataFrame] = None,
        model_predictions: Optional[Dict[str, float]] = None
    ) -> Signal:
        """
        Generate trading signal based on market data and model predictions.

        Args:
            market_data: OHLCV data for the symbol
            features: Optional pre-computed features
            model_predictions: Optional model prediction scores
                Expected keys: 'direction' (1/-1/0), 'confidence' (0-1)

        Returns:
            Signal object with type, strength, and confidence

        Raises:
            ValidationError: If inputs are invalid
            ModelError: If signal generation fails
        """
        try:
            self._validate_market_data(market_data)

            # If no model predictions provided, generate HOLD signal
            if model_predictions is None:
                self.logger.warning(f"No predictions for {self.symbol}, generating HOLD")
                return self._create_hold_signal()

            # Extract prediction components
            direction = model_predictions.get('direction', 0)
            confidence = model_predictions.get('confidence', 0.0)

            # Determine signal type
            if direction > 0:
                signal_type = SignalType.BUY
            elif direction < 0:
                signal_type = SignalType.SELL
            else:
                signal_type = SignalType.HOLD

            # Calculate signal strength based on multiple factors
            strength = self._calculate_strength(
                market_data=market_data,
                features=features,
                direction=direction,
                confidence=confidence
            )

            # Apply filters
            if confidence < self.min_confidence or strength < self.min_strength:
                self.logger.info(
                    f"{self.symbol}: Signal filtered out "
                    f"(confidence={confidence:.2f}, strength={strength:.2f})"
                )
                return self._create_hold_signal()

            # Create signal
            signal = Signal(
                symbol=self.symbol,
                timestamp=datetime.now(),
                signal_type=signal_type,
                strength=strength,
                confidence=confidence,
                metadata={
                    'model_direction': direction,
                    'raw_confidence': confidence,
                    'feature_count': len(features.columns) if features is not None else 0,
                    'market_data_rows': len(market_data)
                }
            )

            self.logger.info(
                f"{self.symbol}: Generated {signal_type.name} signal "
                f"(strength={strength:.2f}, confidence={confidence:.2f})"
            )

            return signal

        except Exception as e:
            self.logger.error(f"Signal generation failed for {self.symbol}: {e}", exc_info=True)
            raise ModelError(f"Failed to generate signal: {e}") from e

    def _validate_market_data(self, df: pd.DataFrame) -> None:
        """Validate market data has required columns."""
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValidationError(f"Market data missing columns: {missing}")
        if len(df) == 0:
            raise ValidationError("Market data is empty")

    def _calculate_strength(
        self,
        market_data: pd.DataFrame,
        features: Optional[pd.DataFrame],
        direction: float,
        confidence: float
    ) -> float:
        """
        Calculate signal strength from multiple factors.

        Signal strength combines:
        - Model confidence
        - Recent volatility (lower is better)
        - Volume confirmation
        """
        # Base strength from model confidence
        strength = abs(direction) * confidence

        # Adjust for volatility (penalize high volatility)
        if len(market_data) >= 20:
            returns = market_data['close'].pct_change().tail(20)
            volatility = returns.std()
            # Normalize volatility (assume 0.02 = average daily vol)
            vol_factor = max(0.5, 1.0 - min(volatility / 0.04, 0.5))
            strength *= vol_factor

        # Adjust for volume (reward above-average volume)
        if len(market_data) >= 20:
            avg_volume = market_data['volume'].tail(20).mean()
            current_volume = market_data['volume'].iloc[-1]
            if avg_volume > 0:
                volume_ratio = min(current_volume / avg_volume, 2.0)
                volume_factor = 0.9 + (volume_ratio - 1.0) * 0.1  # 0.9 to 1.1
                strength *= volume_factor

        # Ensure strength is in valid range
        return min(max(strength, 0.0), 1.0)

    def _create_hold_signal(self) -> Signal:
        """Create a HOLD signal."""
        return Signal(
            symbol=self.symbol,
            timestamp=datetime.now(),
            signal_type=SignalType.HOLD,
            strength=0.0,
            confidence=0.0,
            metadata={'reason': 'default_hold'}
        )


class BatchSignalGenerator:
    """
    Generates signals for multiple symbols in batch.

    This class efficiently processes multiple symbols in parallel and
    aggregates results for portfolio-level decision making.

    Example:
        >>> symbols = ["AAPL", "GOOGL", "MSFT"]
        >>> batch_gen = BatchSignalGenerator(symbols)
        >>> signals = batch_gen.generate_batch(market_data_dict, predictions_dict)
        >>> buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
    """

    def __init__(
        self,
        symbols: List[str],
        min_confidence: float = 0.6,
        min_strength: float = 0.5,
        max_workers: int = 4,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize batch signal generator.

        Args:
            symbols: List of ticker symbols
            min_confidence: Minimum model confidence threshold
            min_strength: Minimum signal strength threshold
            max_workers: Maximum parallel workers for signal generation
            logger: Optional logger instance
        """
        self.symbols = symbols
        self.min_confidence = min_confidence
        self.min_strength = min_strength
        self.max_workers = max_workers
        self.logger = logger or logging.getLogger(__name__)

        # Create individual generators for each symbol
        self.generators = {
            symbol: SignalGenerator(
                symbol=symbol,
                min_confidence=min_confidence,
                min_strength=min_strength,
                logger=self.logger
            )
            for symbol in symbols
        }

        self.logger.info(f"Initialized BatchSignalGenerator for {len(symbols)} symbols")

    def generate_batch(
        self,
        market_data: Dict[str, pd.DataFrame],
        predictions: Dict[str, Dict[str, float]],
        features: Optional[Dict[str, pd.DataFrame]] = None
    ) -> List[Signal]:
        """
        Generate signals for all symbols in batch.

        Args:
            market_data: Dict mapping symbols to their OHLCV data
            predictions: Dict mapping symbols to model predictions
            features: Optional dict mapping symbols to features

        Returns:
            List of Signal objects, one per symbol

        Raises:
            ValidationError: If inputs are invalid
        """
        if not market_data:
            raise ValidationError("market_data cannot be empty")

        signals = []
        errors = []

        for symbol in self.symbols:
            try:
                # Get data for this symbol
                symbol_data = market_data.get(symbol)
                symbol_pred = predictions.get(symbol)
                symbol_features = features.get(symbol) if features else None

                if symbol_data is None:
                    self.logger.warning(f"No market data for {symbol}, skipping")
                    continue

                # Generate signal
                generator = self.generators[symbol]
                signal = generator.generate(
                    market_data=symbol_data,
                    features=symbol_features,
                    model_predictions=symbol_pred
                )
                signals.append(signal)

            except Exception as e:
                self.logger.error(f"Failed to generate signal for {symbol}: {e}")
                errors.append((symbol, str(e)))

        # Log summary
        self.logger.info(
            f"Batch generation complete: {len(signals)} signals, {len(errors)} errors"
        )

        if errors:
            self.logger.warning(f"Failed symbols: {[s for s, _ in errors]}")

        return signals

    def rank_signals(
        self,
        signals: List[Signal],
        top_n: Optional[int] = None
    ) -> List[Tuple[Signal, float]]:
        """
        Rank signals by combined strength and confidence score.

        Args:
            signals: List of signals to rank
            top_n: Optional limit on number of top signals to return

        Returns:
            List of (Signal, score) tuples, sorted by score descending
        """
        # Calculate composite score
        scored_signals = []
        for signal in signals:
            if signal.signal_type == SignalType.HOLD:
                continue

            # Composite score: geometric mean of strength and confidence
            score = np.sqrt(signal.strength * signal.confidence)
            scored_signals.append((signal, score))

        # Sort by score descending
        ranked = sorted(scored_signals, key=lambda x: x[1], reverse=True)

        # Apply top_n limit if specified
        if top_n is not None:
            ranked = ranked[:top_n]

        self.logger.info(f"Ranked {len(ranked)} non-HOLD signals")
        return ranked
