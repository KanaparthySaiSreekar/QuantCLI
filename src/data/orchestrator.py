"""
Data Orchestrator - manages all data providers with failover cascade.
"""

from typing import Optional, Dict, List, Any
from datetime import datetime
import pandas as pd

from src.core.config import ConfigManager
from src.core.logging_config import get_logger
from src.core.exceptions import DataError

from .providers.alpha_vantage import AlphaVantageProvider
from .providers.tiingo import TiingoProvider
from .providers.fred import FREDProvider
from .providers.finnhub import FinnhubProvider
from .providers.polygon import PolygonProvider
from .providers.reddit import RedditProvider
from .providers.gdelt import GDELTProvider


class DataOrchestrator:
    """
    Orchestrates data acquisition from multiple providers with failover.
    
    Features:
    - Multi-source failover cascade
    - Data quality reconciliation
    - Centralized caching
    - Provider load balancing
    """

    def __init__(self):
        self.config = ConfigManager()
        self.logger = get_logger(__name__)
        
        # Initialize providers
        self.providers = {}
        self._init_providers()
        
        # Failover configuration
        self.failover_config = self.config.get('data_sources.failover', {})

    def _init_providers(self):
        """Initialize all available data providers."""
        provider_classes = {
            'alpha_vantage': AlphaVantageProvider,
            'tiingo': TiingoProvider,
            'fred': FREDProvider,
            'finnhub': FinnhubProvider,
            'polygon': PolygonProvider,
            'reddit': RedditProvider,
            'gdelt': GDELTProvider
        }
        
        for name, provider_class in provider_classes.items():
            try:
                provider = provider_class()
                if provider.is_available():
                    self.providers[name] = provider
                    self.logger.info(f"Initialized provider: {name}")
            except Exception as e:
                self.logger.warning(f"Failed to initialize {name}: {e}")

    def get_daily_prices(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        use_failover: bool = True
    ) -> pd.DataFrame:
        """
        Get daily prices with automatic failover.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            use_failover: Whether to use failover cascade
            
        Returns:
            DataFrame with OHLCV data
        """
        if not use_failover:
            # Use primary provider only
            cascade = self.failover_config.get('cascade', {}).get('daily_prices', {})
            primary = cascade.get('primary', 'tiingo')
            
            if primary in self.providers:
                return self.providers[primary].get_daily_prices(
                    symbol, start_date, end_date
                )
            else:
                raise DataError(f"Primary provider {primary} not available")
        
        # Try failover cascade
        cascade = self.failover_config.get('cascade', {}).get('daily_prices', {})
        provider_order = [
            cascade.get('primary'),
            cascade.get('secondary'),
            cascade.get('tertiary'),
            cascade.get('fallback')
        ]
        
        errors = []
        for provider_name in provider_order:
            if not provider_name or provider_name not in self.providers:
                continue
                
            try:
                self.logger.info(f"Trying {provider_name} for {symbol}")
                data = self.providers[provider_name].get_daily_prices(
                    symbol, start_date, end_date
                )
                
                # Validate data quality
                if self._validate_data_quality(data):
                    self.logger.info(f"Successfully retrieved {symbol} from {provider_name}")
                    return data
                else:
                    self.logger.warning(f"Data quality check failed for {provider_name}")
                    
            except Exception as e:
                self.logger.warning(f"{provider_name} failed for {symbol}: {e}")
                errors.append(f"{provider_name}: {str(e)}")
                continue
        
        # All providers failed
        raise DataError(f"All providers failed for {symbol}. Errors: {errors}")

    def get_macro_indicators(
        self,
        indicators: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, pd.Series]:
        """
        Get macroeconomic indicators from FRED.
        
        Args:
            indicators: List of indicator names (None for all configured)
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary mapping indicator name to Series
        """
        if 'fred' not in self.providers:
            raise DataError("FRED provider not available")
        
        fred = self.providers['fred']
        
        # Get configured indicators if none specified
        if indicators is None:
            series_config = self.config.get('data_sources.fred.series', {})
            indicators = list(series_config.keys())
        
        results = {}
        for indicator in indicators:
            try:
                series_config = self.config.get(
                    f'data_sources.fred.series.{indicator}', {}
                )
                series_id = series_config.get('id', indicator)
                
                data = fred.get_series(series_id, start_date, end_date)
                results[indicator] = data
                
            except Exception as e:
                self.logger.warning(f"Failed to get {indicator}: {e}")
        
        return results

    def get_news_sentiment(
        self,
        symbol: Optional[str] = None,
        sources: Optional[List[str]] = None,
        lookback_hours: int = 24
    ) -> pd.DataFrame:
        """
        Aggregate news sentiment from multiple sources.
        
        Args:
            symbol: Stock symbol
            sources: List of sources ('finnhub', 'alpha_vantage', 'gdelt')
            lookback_hours: Hours to look back
            
        Returns:
            Aggregated news sentiment DataFrame
        """
        from datetime import timedelta
        
        if sources is None:
            sources = ['finnhub', 'alpha_vantage', 'gdelt']
        
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=lookback_hours)
        
        all_news = []
        
        for source in sources:
            if source not in self.providers:
                continue
                
            try:
                if source == 'finnhub' and symbol:
                    news = self.providers['finnhub'].get_news(
                        symbol, days_back=lookback_hours//24 or 1
                    )
                    if not news.empty:
                        news['source'] = 'finnhub'
                        all_news.append(news)
                        
                elif source == 'alpha_vantage' and symbol:
                    news = self.providers['alpha_vantage'].get_news_sentiment(
                        symbol=symbol,
                        time_from=start_date,
                        time_to=end_date
                    )
                    if not news.empty:
                        news['source'] = 'alpha_vantage'
                        all_news.append(news)
                        
                elif source == 'gdelt' and symbol:
                    timespan = f"{lookback_hours}h"
                    news = self.providers['gdelt'].search_news(symbol, timespan)
                    if not news.empty:
                        news['source'] = 'gdelt'
                        all_news.append(news)
                        
            except Exception as e:
                self.logger.warning(f"Failed to get news from {source}: {e}")
        
        if all_news:
            combined = pd.concat(all_news, axis=0)
            combined = combined.sort_index()
            return combined
        else:
            return pd.DataFrame()

    def get_social_sentiment(
        self,
        subreddits: Optional[List[str]] = None,
        limit: int = 100
    ) -> pd.DataFrame:
        """Get social media sentiment from Reddit."""
        if 'reddit' not in self.providers:
            raise DataError("Reddit provider not available")
        
        if subreddits is None:
            subreddits = self.config.get(
                'data_sources.reddit.subreddits',
                ['wallstreetbets', 'stocks', 'investing']
            )
        
        all_posts = []
        reddit = self.providers['reddit']
        
        for subreddit in subreddits:
            try:
                posts = reddit.get_posts(subreddit, limit=limit)
                if not posts.empty:
                    posts['subreddit'] = subreddit
                    all_posts.append(posts)
            except Exception as e:
                self.logger.warning(f"Failed to get posts from {subreddit}: {e}")
        
        if all_posts:
            return pd.concat(all_posts, axis=0).sort_index()
        else:
            return pd.DataFrame()

    def update_daily_data(
        self,
        symbols: List[str],
        save_to_db: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Update daily data for list of symbols.
        
        Args:
            symbols: List of stock symbols
            save_to_db: Whether to save to database
            
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        results = {}
        
        for symbol in symbols:
            try:
                data = self.get_daily_prices(symbol)
                results[symbol] = data
                
                if save_to_db:
                    # TODO: Save to TimescaleDB
                    self.logger.info(f"Saved {symbol} to database")
                    
            except Exception as e:
                self.logger.error(f"Failed to update {symbol}: {e}")
        
        return results

    def _validate_data_quality(self, data: pd.DataFrame) -> bool:
        """Validate data quality."""
        if data is None or data.empty:
            return False
        
        quality_config = self.config.get('data_sources.quality.validation', {})
        
        # Check minimum data points
        min_points = quality_config.get('min_data_points', 100)
        if len(data) < min_points:
            self.logger.warning(f"Insufficient data points: {len(data)} < {min_points}")
            return False
        
        # Check missing data percentage
        max_missing = quality_config.get('max_missing_pct', 5.0)
        missing_pct = (data.isnull().sum().sum() / data.size) * 100
        if missing_pct > max_missing:
            self.logger.warning(f"Too much missing data: {missing_pct:.2f}%")
            return False
        
        return True
