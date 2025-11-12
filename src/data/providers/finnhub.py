"""
Finnhub data provider.
60 calls/minute free tier.
"""

from datetime import datetime
from typing import Optional, Dict, List
import pandas as pd
import finnhub

from .base import BaseDataProvider
from src.core.exceptions import DataError


class FinnhubProvider(BaseDataProvider):
    """Finnhub provider for news, sentiment, and quotes."""

    def __init__(self):
        super().__init__('finnhub')
        
        if not self.api_key:
            raise DataError("Finnhub API key not configured")
            
        # Initialize Finnhub client
        self.client = finnhub.Client(api_key=self.api_key)

    def is_available(self) -> bool:
        return bool(self.api_key) and self.provider_config.get('enabled', False)

    def get_daily_prices(self, symbol: str, start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Get daily OHLCV data from Finnhub."""
        cache_key = self._cache_key('daily', symbol, start_date, end_date)
        
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data

        try:
            # Convert dates to timestamps
            start_ts = int(start_date.timestamp()) if start_date else None
            end_ts = int(end_date.timestamp()) if end_date else None
            
            # Get candle data
            data = self.client.stock_candles(symbol, 'D', start_ts, end_ts)
            
            if data['s'] != 'ok':
                raise DataError(f"Finnhub returned status: {data['s']}")
            
            # Convert to DataFrame
            df = pd.DataFrame({
                'open': data['o'],
                'high': data['h'],
                'low': data['l'],
                'close': data['c'],
                'volume': data['v']
            }, index=pd.to_datetime(data['t'], unit='s'))
            
            df = df.sort_index()
            
            # Cache result
            self._save_to_cache(cache_key, df)
            
            return df

        except Exception as e:
            raise DataError(f"Finnhub error for {symbol}: {e}")

    def get_news(self, symbol: str, days_back: int = 7) -> pd.DataFrame:
        """Get company news."""
        from datetime import timedelta
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        cache_key = self._cache_key('news', symbol, start_date, end_date)
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data

        try:
            news = self.client.company_news(
                symbol,
                _from=start_date.strftime('%Y-%m-%d'),
                to=end_date.strftime('%Y-%m-%d')
            )
            
            df = pd.DataFrame(news)
            if not df.empty:
                df['datetime'] = pd.to_datetime(df['datetime'], unit='s')
                df = df.set_index('datetime').sort_index()
            
            self._save_to_cache(cache_key, df)
            return df

        except Exception as e:
            raise DataError(f"Finnhub news error for {symbol}: {e}")
