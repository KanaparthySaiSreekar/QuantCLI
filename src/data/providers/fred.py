"""
FRED (Federal Reserve Economic Data) provider.
Unlimited calls for macroeconomic indicators.
"""

from datetime import datetime
from typing import Optional, List
import pandas as pd
from fredapi import Fred

from .base import BaseDataProvider
from src.core.exceptions import DataError


class FREDProvider(BaseDataProvider):
    """FRED data provider for macroeconomic indicators."""

    def __init__(self):
        super().__init__('fred')
        
        if not self.api_key:
            raise DataError("FRED API key not configured")
            
        # Initialize FRED client
        self.client = Fred(api_key=self.api_key)

    def is_available(self) -> bool:
        return bool(self.api_key) and self.provider_config.get('enabled', False)

    def get_daily_prices(self, symbol: str, start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Not applicable for FRED - use get_series instead."""
        raise NotImplementedError("Use get_series() for FRED data")

    def get_series(
        self,
        series_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.Series:
        """
        Get economic data series.
        
        Args:
            series_id: FRED series ID (e.g., 'VIXCLS', 'DGS10')
            start_date: Start date
            end_date: End date
            
        Returns:
            Pandas Series with data
        """
        cache_key = self._cache_key('series', series_id, start_date, end_date)
        
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data

        try:
            data = self.client.get_series(
                series_id,
                observation_start=start_date,
                observation_end=end_date
            )
            
            # Cache result
            self._save_to_cache(cache_key, data)
            
            self.logger.info(f"Retrieved {len(data)} observations for {series_id}")
            return data

        except Exception as e:
            raise DataError(f"FRED error for {series_id}: {e}")

    def get_vix(self, start_date: Optional[datetime] = None,
                end_date: Optional[datetime] = None) -> pd.Series:
        """Get VIX (volatility index) data."""
        return self.get_series('VIXCLS', start_date, end_date)

    def get_treasury_yield(self, maturity: str = '10Y',
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None) -> pd.Series:
        """Get Treasury yield data."""
        series_mapping = {
            '1M': 'DGS1MO',
            '3M': 'DGS3MO',
            '6M': 'DGS6MO',
            '1Y': 'DGS1',
            '2Y': 'DGS2',
            '5Y': 'DGS5',
            '10Y': 'DGS10',
            '20Y': 'DGS20',
            '30Y': 'DGS30'
        }
        series_id = series_mapping.get(maturity, 'DGS10')
        return self.get_series(series_id, start_date, end_date)
