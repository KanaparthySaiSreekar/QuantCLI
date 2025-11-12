"""
GDELT data provider for global news.
"""

from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import gdelt

from .base import BaseDataProvider
from src.core.exceptions import DataError


class GDELTProvider(BaseDataProvider):
    """GDELT provider for global news coverage."""

    def __init__(self):
        super().__init__('gdelt')
        self.gd = gdelt.gdelt(version=2)

    def is_available(self) -> bool:
        return self.provider_config.get('enabled', False)

    def get_daily_prices(self, symbol: str, start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Not applicable for GDELT."""
        raise NotImplementedError("Use search_news()")

    def search_news(self, query: str, timespan: str = '1d',
                   max_results: int = 250) -> pd.DataFrame:
        """Search GDELT for news articles."""
        cache_key = self._cache_key('search', query, timespan, max_results)
        
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data

        try:
            # GDELT search
            results = self.gd.Search(
                query,
                table='gkg',
                coverage=True,
                timespan=timespan
            )
            
            if results is not None and not results.empty:
                results = results.head(max_results)
                self._save_to_cache(cache_key, results)
                return results
            else:
                return pd.DataFrame()

        except Exception as e:
            self.logger.warning(f"GDELT search error: {e}")
            return pd.DataFrame()
