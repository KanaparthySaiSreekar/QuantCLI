"""
Tiingo data provider.
50 symbols/hour, 30+ years historical data.
"""

from datetime import datetime
from typing import Optional
import pandas as pd
from tiingo import TiingoClient

from .base import BaseDataProvider
from src.core.exceptions import DataError


class TiingoProvider(BaseDataProvider):
    """Tiingo data provider for historical EOD data."""

    def __init__(self):
        super().__init__('tiingo')
        
        if not self.api_key:
            raise DataError("Tiingo API key not configured")
            
        # Initialize Tiingo client
        config = {
            'api_key': self.api_key,
            'session': True
        }
        self.client = TiingoClient(config)

    def is_available(self) -> bool:
        return bool(self.api_key) and self.provider_config.get('enabled', False)

    def get_daily_prices(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Get daily OHLCV data with 30+ years history."""
        cache_key = self._cache_key('daily', symbol, start_date, end_date)
        
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data

        try:
            # Build parameters
            params = {}
            if start_date:
                params['startDate'] = start_date.strftime('%Y-%m-%d')
            if end_date:
                params['endDate'] = end_date.strftime('%Y-%m-%d')

            # Get data
            data = self.client.get_dataframe(symbol, **params)
            
            # Standardize column names
            data = data.rename(columns={
                'adjClose': 'adjusted_close',
                'adjHigh': 'adjusted_high',
                'adjLow': 'adjusted_low',
                'adjOpen': 'adjusted_open',
                'adjVolume': 'adjusted_volume',
                'divCash': 'dividend',
                'splitFactor': 'split_coefficient'
            })
            
            # Ensure datetime index
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
            
            data = data.sort_index()
            
            # Cache result
            self._save_to_cache(cache_key, data)
            
            self.logger.info(f"Retrieved {len(data)} daily prices for {symbol} from Tiingo")
            return data

        except Exception as e:
            raise DataError(f"Tiingo error for {symbol}: {e}")
