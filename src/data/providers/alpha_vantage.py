"""
Alpha Vantage data provider.
25 calls/day, 5 calls/minute free tier.
"""

from datetime import datetime
from typing import Optional, Dict, List
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData

from .base import BaseDataProvider
from src.core.exceptions import DataError


class AlphaVantageProvider(BaseDataProvider):
    """
    Alpha Vantage data provider.

    Provides:
    - Daily/intraday stock prices
    - News sentiment
    - Fundamental data
    """

    def __init__(self):
        super().__init__('alpha_vantage')

        if not self.api_key:
            raise DataError("Alpha Vantage API key not configured")

        # Initialize Alpha Vantage clients
        self.ts = TimeSeries(key=self.api_key, output_format='pandas')
        self.fd = FundamentalData(key=self.api_key, output_format='pandas')

    def is_available(self) -> bool:
        """Check if provider is available."""
        return bool(self.api_key) and self.provider_config.get('enabled', False)

    def get_daily_prices(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        adjusted: bool = True
    ) -> pd.DataFrame:
        """
        Get daily OHLCV data.

        Args:
            symbol: Stock symbol
            start_date: Start date (if None, get full history)
            end_date: End date (if None, use today)
            adjusted: Whether to use adjusted prices

        Returns:
            DataFrame with columns: open, high, low, close, volume
        """
        cache_key = self._cache_key('daily', symbol, start_date, end_date, adjusted)

        # Check cache
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data

        try:
            if adjusted:
                data, meta_data = self.ts.get_daily_adjusted(
                    symbol=symbol,
                    outputsize='full'
                )
            else:
                data, meta_data = self.ts.get_daily(
                    symbol=symbol,
                    outputsize='full'
                )

            # Rename columns
            column_mapping = {
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close',
                '5. adjusted close': 'adjusted_close',
                '5. volume': 'volume',
                '6. volume': 'volume',
                '7. dividend amount': 'dividend',
                '8. split coefficient': 'split_coefficient'
            }

            data = data.rename(columns=column_mapping)

            # Convert index to datetime
            data.index = pd.to_datetime(data.index)
            data = data.sort_index()

            # Filter by date range
            if start_date:
                data = data[data.index >= start_date]
            if end_date:
                data = data[data.index <= end_date]

            # Select relevant columns
            columns = ['open', 'high', 'low', 'close', 'volume']
            if adjusted and 'adjusted_close' in data.columns:
                columns.append('adjusted_close')

            data = data[columns]

            # Cache result
            self._save_to_cache(cache_key, data)

            self.logger.info(f"Retrieved {len(data)} daily prices for {symbol}")
            return data

        except Exception as e:
            self.logger.error(f"Failed to get daily prices for {symbol}: {e}")
            raise DataError(f"Alpha Vantage error: {e}")

    def get_quote(self, symbol: str) -> Dict:
        """
        Get latest quote for symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with quote data
        """
        cache_key = self._cache_key('quote', symbol)

        # Check cache (shorter TTL for quotes)
        if cache_key in self.cache:
            data, timestamp = self.cache[cache_key]
            import time
            if time.time() - timestamp < 60:  # 1 minute cache
                return data

        try:
            data, meta_data = self.ts.get_quote_endpoint(symbol=symbol)

            quote = {
                'symbol': data['01. symbol'].iloc[0],
                'price': float(data['05. price'].iloc[0]),
                'volume': int(data['06. volume'].iloc[0]),
                'latest_trading_day': data['07. latest trading day'].iloc[0],
                'change': float(data['09. change'].iloc[0]),
                'change_percent': data['10. change percent'].iloc[0]
            }

            # Cache with short TTL
            import time
            self.cache[cache_key] = (quote, time.time())

            return quote

        except Exception as e:
            raise DataError(f"Failed to get quote for {symbol}: {e}")

    def get_news_sentiment(
        self,
        symbol: Optional[str] = None,
        topics: Optional[List[str]] = None,
        time_from: Optional[datetime] = None,
        time_to: Optional[datetime] = None,
        limit: int = 50
    ) -> pd.DataFrame:
        """
        Get news sentiment data.

        Args:
            symbol: Stock symbol (optional)
            topics: List of topics (optional)
            time_from: Start datetime
            time_to: End datetime
            limit: Maximum number of articles

        Returns:
            DataFrame with news sentiment
        """
        params = {
            'function': 'NEWS_SENTIMENT',
            'apikey': self.api_key,
            'limit': limit
        }

        if symbol:
            params['tickers'] = symbol
        if topics:
            params['topics'] = ','.join(topics)
        if time_from:
            params['time_from'] = time_from.strftime('%Y%m%dT%H%M')
        if time_to:
            params['time_to'] = time_to.strftime('%Y%m%dT%H%M')

        try:
            response = self._make_request('', params=params)
            data = response.json()

            if 'feed' not in data:
                raise DataError("No news data returned")

            # Convert to DataFrame
            articles = []
            for item in data['feed']:
                article = {
                    'title': item.get('title'),
                    'url': item.get('url'),
                    'time_published': pd.to_datetime(item.get('time_published')),
                    'summary': item.get('summary'),
                    'source': item.get('source'),
                    'overall_sentiment_score': float(item.get('overall_sentiment_score', 0)),
                    'overall_sentiment_label': item.get('overall_sentiment_label')
                }

                # Add ticker-specific sentiment if symbol provided
                if symbol and 'ticker_sentiment' in item:
                    for ticker_sent in item['ticker_sentiment']:
                        if ticker_sent.get('ticker') == symbol:
                            article['ticker_sentiment_score'] = float(
                                ticker_sent.get('ticker_sentiment_score', 0)
                            )
                            article['ticker_sentiment_label'] = ticker_sent.get(
                                'ticker_sentiment_label'
                            )
                            article['relevance_score'] = float(
                                ticker_sent.get('relevance_score', 0)
                            )

                articles.append(article)

            df = pd.DataFrame(articles)
            df = df.set_index('time_published').sort_index()

            self.logger.info(f"Retrieved {len(df)} news articles")
            return df

        except Exception as e:
            raise DataError(f"Failed to get news sentiment: {e}")

    def get_company_overview(self, symbol: str) -> Dict:
        """
        Get fundamental company data.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with company fundamentals
        """
        cache_key = self._cache_key('overview', symbol)

        # Check cache (daily TTL for fundamentals)
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data

        try:
            data, meta_data = self.fd.get_company_overview(symbol=symbol)

            overview = data.to_dict('records')[0] if not data.empty else {}

            # Cache result
            self._save_to_cache(cache_key, overview)

            return overview

        except Exception as e:
            raise DataError(f"Failed to get company overview for {symbol}: {e}")
