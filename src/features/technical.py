"""
Technical indicators for feature engineering.

Implements common technical analysis indicators:
- Trend indicators (SMA, EMA, MACD)
- Momentum indicators (RSI, Stochastic, ROC)
- Volatility indicators (Bollinger Bands, ATR)
- Volume indicators (OBV, VWAP)
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
from src.core.logging_config import get_logger

logger = get_logger(__name__)


class TechnicalIndicators:
    """
    Calculate technical indicators from OHLCV data.

    All methods are staticmethods for easy reuse and testing.
    """

    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """
        Simple Moving Average.

        Args:
            series: Price series
            period: Number of periods

        Returns:
            SMA series
        """
        return series.rolling(window=period).mean()

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """
        Exponential Moving Average.

        Args:
            series: Price series
            period: Number of periods

        Returns:
            EMA series
        """
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index.

        Args:
            series: Price series
            period: RSI period (default 14)

        Returns:
            RSI series (0-100)
        """
        delta = series.diff()

        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def macd(
        series: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Moving Average Convergence Divergence.

        Args:
            series: Price series
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period

        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        ema_fast = series.ewm(span=fast_period, adjust=False).mean()
        ema_slow = series.ewm(span=slow_period, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    @staticmethod
    def bollinger_bands(
        series: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands.

        Args:
            series: Price series
            period: Moving average period
            std_dev: Number of standard deviations

        Returns:
            Tuple of (Upper band, Middle band, Lower band)
        """
        middle_band = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()

        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)

        return upper_band, middle_band, lower_band

    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Average True Range.

        Args:
            df: OHLCV DataFrame
            period: ATR period

        Returns:
            ATR series
        """
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(span=period, adjust=False).mean()

        return atr

    @staticmethod
    def stochastic(
        df: pd.DataFrame,
        period: int = 14,
        smooth_k: int = 3,
        smooth_d: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Stochastic Oscillator.

        Args:
            df: OHLCV DataFrame
            period: Lookback period
            smooth_k: %K smoothing period
            smooth_d: %D smoothing period

        Returns:
            Tuple of (%K, %D)
        """
        high = df['high']
        low = df['low']
        close = df['close']

        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()

        raw_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        k = raw_k.rolling(window=smooth_k).mean()
        d = k.rolling(window=smooth_d).mean()

        return k, d

    @staticmethod
    def obv(df: pd.DataFrame) -> pd.Series:
        """
        On-Balance Volume.

        Args:
            df: OHLCV DataFrame

        Returns:
            OBV series
        """
        price_change = df['close'].diff()
        volume = df['volume']

        obv = np.where(price_change > 0, volume,
               np.where(price_change < 0, -volume, 0))

        return pd.Series(obv, index=df.index).cumsum()

    @staticmethod
    def vwap(df: pd.DataFrame) -> pd.Series:
        """
        Volume Weighted Average Price.

        Args:
            df: OHLCV DataFrame with timestamp index

        Returns:
            VWAP series
        """
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()

        return vwap

    @staticmethod
    def roc(series: pd.Series, period: int = 12) -> pd.Series:
        """
        Rate of Change.

        Args:
            series: Price series
            period: ROC period

        Returns:
            ROC series (percentage)
        """
        roc = ((series - series.shift(period)) / series.shift(period)) * 100
        return roc

    @staticmethod
    def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Williams %R.

        Args:
            df: OHLCV DataFrame
            period: Lookback period

        Returns:
            Williams %R series (-100 to 0)
        """
        high = df['high']
        low = df['low']
        close = df['close']

        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()

        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)

        return williams_r

    @staticmethod
    def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        Commodity Channel Index.

        Args:
            df: OHLCV DataFrame
            period: CCI period

        Returns:
            CCI series
        """
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(
            lambda x: np.abs(x - x.mean()).mean()
        )

        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)

        return cci

    @staticmethod
    def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Average Directional Index.

        Args:
            df: OHLCV DataFrame
            period: ADX period

        Returns:
            ADX series
        """
        high = df['high']
        low = df['low']
        close = df['close']

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        up_move = high - high.shift()
        down_move = low.shift() - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # Smooth the values
        atr = tr.ewm(span=period, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(span=period, adjust=False).mean() / atr
        minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(span=period, adjust=False).mean() / atr

        # Calculate ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(span=period, adjust=False).mean()

        return adx

    @staticmethod
    def money_flow_index(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Money Flow Index.

        Args:
            df: OHLCV DataFrame
            period: MFI period

        Returns:
            MFI series (0-100)
        """
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']

        # Positive and negative money flow
        price_change = typical_price.diff()
        positive_flow = money_flow.where(price_change > 0, 0)
        negative_flow = money_flow.where(price_change < 0, 0)

        # Money flow ratio
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()

        mfr = positive_mf / negative_mf
        mfi = 100 - (100 / (1 + mfr))

        return mfi

    @staticmethod
    def keltner_channels(
        df: pd.DataFrame,
        ema_period: int = 20,
        atr_period: int = 10,
        multiplier: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Keltner Channels.

        Args:
            df: OHLCV DataFrame
            ema_period: EMA period for middle line
            atr_period: ATR period
            multiplier: ATR multiplier for bands

        Returns:
            Tuple of (Upper channel, Middle line, Lower channel)
        """
        middle_line = df['close'].ewm(span=ema_period, adjust=False).mean()
        atr = TechnicalIndicators.atr(df, period=atr_period)

        upper_channel = middle_line + (multiplier * atr)
        lower_channel = middle_line - (multiplier * atr)

        return upper_channel, middle_line, lower_channel

    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators at once.

        Args:
            df: OHLCV DataFrame

        Returns:
            DataFrame with all indicators added as columns
        """
        result = df.copy()

        # Moving averages
        for period in [5, 10, 20, 50, 200]:
            result[f'sma_{period}'] = TechnicalIndicators.sma(df['close'], period)
            result[f'ema_{period}'] = TechnicalIndicators.ema(df['close'], period)

        # Momentum indicators
        result['rsi_14'] = TechnicalIndicators.rsi(df['close'], 14)
        result['roc_12'] = TechnicalIndicators.roc(df['close'], 12)

        # MACD
        macd, signal, hist = TechnicalIndicators.macd(df['close'])
        result['macd'] = macd
        result['macd_signal'] = signal
        result['macd_hist'] = hist

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(df['close'])
        result['bb_upper'] = bb_upper
        result['bb_middle'] = bb_middle
        result['bb_lower'] = bb_lower

        # Volatility
        result['atr_14'] = TechnicalIndicators.atr(df, 14)

        # Stochastic
        stoch_k, stoch_d = TechnicalIndicators.stochastic(df)
        result['stoch_k'] = stoch_k
        result['stoch_d'] = stoch_d

        # Volume indicators
        result['obv'] = TechnicalIndicators.obv(df)
        result['vwap'] = TechnicalIndicators.vwap(df)
        result['mfi_14'] = TechnicalIndicators.money_flow_index(df, 14)

        # Other indicators
        result['williams_r_14'] = TechnicalIndicators.williams_r(df, 14)
        result['cci_20'] = TechnicalIndicators.cci(df, 20)
        result['adx_14'] = TechnicalIndicators.adx(df, 14)

        logger.info(f"Calculated {len(result.columns) - len(df.columns)} technical indicators")

        return result
