"""
Feature engineering orchestration.

Combines multiple feature types:
- Technical indicators
- Price-based features
- Volume-based features
- Time-based features
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from datetime import datetime

from .technical import TechnicalIndicators
from src.core.logging_config import get_logger
from src.core.exceptions import DataError

logger = get_logger(__name__)


class FeatureEngineer:
    """
    Orchestrates feature engineering for ML models.

    Generates comprehensive feature set including:
    - Technical indicators
    - Price transformations
    - Volume analysis
    - Time-based features
    """

    def __init__(
        self,
        include_technical: bool = True,
        include_price: bool = True,
        include_volume: bool = True,
        include_time: bool = True
    ):
        """
        Initialize FeatureEngineer.

        Args:
            include_technical: Include technical indicators
            include_price: Include price-based features
            include_volume: Include volume-based features
            include_time: Include time-based features
        """
        self.include_technical = include_technical
        self.include_price = include_price
        self.include_volume = include_volume
        self.include_time = include_time
        self.logger = logger

    def generate_features(
        self,
        df: pd.DataFrame,
        dropna: bool = True
    ) -> pd.DataFrame:
        """
        Generate all enabled features.

        Args:
            df: OHLCV DataFrame with datetime index
            dropna: Drop rows with NaN values

        Returns:
            DataFrame with all features

        Raises:
            DataError: If input data is invalid
        """
        self._validate_input(df)

        result = df.copy()

        if self.include_technical:
            result = self._add_technical_features(result)

        if self.include_price:
            result = self._add_price_features(result)

        if self.include_volume:
            result = self._add_volume_features(result)

        if self.include_time:
            result = self._add_time_features(result)

        # Drop NaN values if requested
        if dropna:
            rows_before = len(result)
            result = result.dropna()
            rows_after = len(result)
            if rows_before - rows_after > 0:
                self.logger.info(f"Dropped {rows_before - rows_after} rows with NaN values")

        feature_count = len(result.columns) - len(df.columns)
        self.logger.info(f"Generated {feature_count} features, {len(result)} rows")

        return result

    def _validate_input(self, df: pd.DataFrame) -> None:
        """Validate input DataFrame."""
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_cols if col not in df.columns]

        if missing:
            raise DataError(f"Missing required columns: {missing}")

        if len(df) == 0:
            raise DataError("Input DataFrame is empty")

        if not isinstance(df.index, pd.DatetimeIndex):
            raise DataError("DataFrame index must be DatetimeIndex")

    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicator features."""
        self.logger.debug("Adding technical indicator features")

        # Use the comprehensive technical indicators
        result = TechnicalIndicators.calculate_all_indicators(df)

        # Add custom combinations
        # Price position within Bollinger Bands
        if 'bb_upper' in result.columns and 'bb_lower' in result.columns:
            bb_width = result['bb_upper'] - result['bb_lower']
            result['bb_position'] = (result['close'] - result['bb_lower']) / bb_width
            result['bb_width_pct'] = bb_width / result['close']

        # MACD crossover signal
        if 'macd' in result.columns and 'macd_signal' in result.columns:
            result['macd_cross'] = np.where(
                result['macd'] > result['macd_signal'], 1, -1
            )

        # Trend strength
        if 'sma_50' in result.columns and 'sma_200' in result.columns:
            result['trend_strength'] = (result['sma_50'] - result['sma_200']) / result['sma_200']

        return result

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""
        self.logger.debug("Adding price-based features")

        result = df.copy()

        # Returns over different periods
        for period in [1, 2, 3, 5, 10, 20]:
            result[f'return_{period}d'] = result['close'].pct_change(period)

        # Log returns
        result['log_return_1d'] = np.log(result['close'] / result['close'].shift(1))

        # Intraday range
        result['intraday_range'] = (result['high'] - result['low']) / result['close']

        # Gap (open vs previous close)
        result['gap'] = (result['open'] - result['close'].shift(1)) / result['close'].shift(1)

        # High-Low position
        result['hl_position'] = (result['close'] - result['low']) / (result['high'] - result['low'])

        # Price acceleration (second derivative)
        result['price_acceleration'] = result['close'].diff().diff()

        # Distance from highs/lows
        for period in [20, 50]:
            result[f'dist_from_high_{period}'] = (
                result['close'] / result['high'].rolling(period).max() - 1
            )
            result[f'dist_from_low_{period}'] = (
                result['close'] / result['low'].rolling(period).min() - 1
            )

        # Volatility measures
        for period in [5, 10, 20]:
            result[f'volatility_{period}d'] = result['return_1d'].rolling(period).std()

        return result

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        self.logger.debug("Adding volume-based features")

        result = df.copy()

        # Volume moving averages
        for period in [5, 10, 20, 50]:
            result[f'volume_sma_{period}'] = result['volume'].rolling(period).mean()

        # Volume ratio (current vs average)
        result['volume_ratio_20'] = result['volume'] / result['volume_sma_20']

        # Volume changes
        result['volume_change_1d'] = result['volume'].pct_change(1)

        # Price-volume correlation
        for period in [10, 20]:
            result[f'price_volume_corr_{period}'] = result['close'].rolling(period).corr(
                result['volume']
            )

        # Volume trend
        result['volume_trend_20'] = (
            result['volume_sma_5'] / result['volume_sma_20']
        )

        # Accumulation/Distribution Line
        mfm = ((result['close'] - result['low']) - (result['high'] - result['close'])) / (
            result['high'] - result['low']
        )
        mfv = mfm * result['volume']
        result['ad_line'] = mfv.cumsum()

        return result

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        self.logger.debug("Adding time-based features")

        result = df.copy()

        # Extract datetime components
        result['day_of_week'] = result.index.dayofweek  # Monday=0, Sunday=6
        result['day_of_month'] = result.index.day
        result['month'] = result.index.month
        result['quarter'] = result.index.quarter
        result['week_of_year'] = result.index.isocalendar().week

        # Binary features for specific days
        result['is_monday'] = (result['day_of_week'] == 0).astype(int)
        result['is_friday'] = (result['day_of_week'] == 4).astype(int)
        result['is_month_end'] = result.index.is_month_end.astype(int)
        result['is_month_start'] = result.index.is_month_start.astype(int)
        result['is_quarter_end'] = result.index.is_quarter_end.astype(int)

        # Cyclical encoding for day of week and month
        result['day_of_week_sin'] = np.sin(2 * np.pi * result['day_of_week'] / 7)
        result['day_of_week_cos'] = np.cos(2 * np.pi * result['day_of_week'] / 7)
        result['month_sin'] = np.sin(2 * np.pi * result['month'] / 12)
        result['month_cos'] = np.cos(2 * np.pi * result['month'] / 12)

        return result

    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of feature column names (excluding OHLCV).

        Args:
            df: DataFrame with features

        Returns:
            List of feature column names
        """
        base_cols = ['open', 'high', 'low', 'close', 'volume']
        feature_cols = [col for col in df.columns if col not in base_cols]
        return feature_cols

    def select_features(
        self,
        df: pd.DataFrame,
        method: str = 'variance',
        n_features: Optional[int] = None,
        threshold: float = 0.01
    ) -> List[str]:
        """
        Select most important features.

        Args:
            df: DataFrame with all features
            method: Selection method ('variance', 'correlation')
            n_features: Number of features to select (None = use threshold)
            threshold: Threshold for selection

        Returns:
            List of selected feature names
        """
        feature_cols = self.get_feature_names(df)

        if method == 'variance':
            # Remove low-variance features
            variances = df[feature_cols].var()
            if n_features:
                selected = variances.nlargest(n_features).index.tolist()
            else:
                selected = variances[variances > threshold].index.tolist()

        elif method == 'correlation':
            # Remove highly correlated features
            corr_matrix = df[feature_cols].corr().abs()
            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
            selected = [col for col in feature_cols if col not in to_drop]

        else:
            raise ValueError(f"Unknown selection method: {method}")

        self.logger.info(
            f"Selected {len(selected)}/{len(feature_cols)} features using {method}"
        )

        return selected

    def transform_for_ml(
        self,
        df: pd.DataFrame,
        target_periods: int = 1,
        classification: bool = False
    ) -> pd.DataFrame:
        """
        Transform features for ML model training.

        Args:
            df: DataFrame with features
            target_periods: Number of periods ahead for target
            classification: Create classification target (up/down)

        Returns:
            DataFrame with features and target column
        """
        result = df.copy()

        # Create target variable (future returns)
        future_return = result['close'].pct_change(target_periods).shift(-target_periods)

        if classification:
            # Binary classification: 1 = up, 0 = down
            result['target'] = (future_return > 0).astype(int)
        else:
            # Regression: actual return
            result['target'] = future_return

        # Remove rows without target
        result = result.dropna(subset=['target'])

        self.logger.info(
            f"Created {'classification' if classification else 'regression'} "
            f"target with {len(result)} samples"
        )

        return result
