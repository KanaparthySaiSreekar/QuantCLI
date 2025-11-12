"""
Polygon.io data provider.
5 calls/minute free tier, 2 years historical.
"""

from datetime import datetime
from typing import Optional
import pandas as pd
from polygon import RESTClient

from .base import BaseDataProvider
from src.core.exceptions import DataError


class PolygonProvider(BaseDataProvider):
    """Polygon.io provider - high quality EOD data."""

    def __init__(self):
        super().__init__('polygon')
        
        if not self.api_key:
            raise DataError("Polygon API key not configured")
            
        self.client = RESTClient(api_key=self.api_key)

    def is_available(self) -> bool:
        return bool(self.api_key) and self.provider_config.get('enabled', False)

    def get_daily_prices(self, symbol: str, start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Get daily OHLCV data."""
        cache_key = self._cache_key('daily', symbol, start_date, end_date)
        
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data

        try:
            aggs = []
            for agg in self.client.list_aggs(
                ticker=symbol,
                multiplier=1,
                timespan="day",
                from_=start_date.strftime('%Y-%m-%d') if start_date else None,
                to=end_date.strftime('%Y-%m-%d') if end_date else None
            ):
                aggs.append(agg)
            
            if not aggs:
                raise DataError(f"No data returned for {symbol}")
            
            # Convert to DataFrame
            df = pd.DataFrame([{
                'open': a.open,
                'high': a.high,
                'low': a.low,
                'close': a.close,
                'volume': a.volume,
                'vwap': a.vwap,
                'timestamp': a.timestamp
            } for a in aggs])
            
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('date').sort_index()
            df = df[['open', 'high', 'low', 'close', 'volume', 'vwap']]
            
            self._save_to_cache(cache_key, df)
            return df

        except Exception as e:
            raise DataError(f"Polygon error for {symbol}: {e}")
