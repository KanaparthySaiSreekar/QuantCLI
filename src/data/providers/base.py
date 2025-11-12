"""
Base data provider class with rate limiting, caching, and retry logic.
"""

import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
from functools import wraps
import hashlib
import json

from src.core.config import ConfigManager
from src.core.logging_config import get_logger
from src.core.exceptions import DataError, RateLimitError

logger = get_logger(__name__)


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, calls: int, period: int):
        """
        Initialize rate limiter.

        Args:
            calls: Number of calls allowed
            period: Time period in seconds
        """
        self.calls = calls
        self.period = period
        self.tokens = calls
        self.last_update = time.time()

    def acquire(self, blocking: bool = True) -> bool:
        """
        Acquire a token for making a request.

        Args:
            blocking: If True, wait until token available

        Returns:
            True if token acquired, False otherwise
        """
        now = time.time()
        elapsed = now - self.last_update

        # Refill tokens based on elapsed time
        self.tokens = min(
            self.calls,
            self.tokens + (elapsed * self.calls / self.period)
        )
        self.last_update = now

        if self.tokens >= 1:
            self.tokens -= 1
            return True
        elif blocking:
            # Wait until next token available
            sleep_time = (1 - self.tokens) * self.period / self.calls
            logger.debug(f"Rate limit reached, sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
            self.tokens = 0
            self.last_update = time.time()
            return True
        else:
            return False


class BaseDataProvider(ABC):
    """
    Abstract base class for data providers.

    Provides:
    - Rate limiting
    - Caching
    - Retry logic with exponential backoff
    - Error handling
    - Request session management
    """

    def __init__(self, config_key: str):
        """
        Initialize data provider.

        Args:
            config_key: Key in config file (e.g., 'alpha_vantage')
        """
        self.config = ConfigManager()
        self.config_key = config_key
        self.provider_config = self.config.get(f'data_sources.{config_key}', {})

        # Setup logging
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")

        # Rate limiting
        rate_limits = self.provider_config.get('rate_limits', {})
        self.rate_limiters = {}

        if 'calls_per_day' in rate_limits:
            self.rate_limiters['daily'] = RateLimiter(
                rate_limits['calls_per_day'],
                86400  # 24 hours in seconds
            )

        if 'calls_per_minute' in rate_limits:
            self.rate_limiters['minute'] = RateLimiter(
                rate_limits['calls_per_minute'],
                60
            )

        if 'symbols_per_hour' in rate_limits:
            self.rate_limiters['hourly'] = RateLimiter(
                rate_limits['symbols_per_hour'],
                3600
            )

        # Retry configuration
        retry_config = self.provider_config.get('retry', {})
        self.max_retries = retry_config.get('max_attempts', 3)
        self.backoff_factor = retry_config.get('backoff_factor', 2)
        self.backoff_max = retry_config.get('backoff_max', 60)

        # Cache configuration
        self.cache_ttl = self.provider_config.get('cache_ttl_seconds', 3600)
        self.cache = {}

        # Setup HTTP session with retry logic
        self.session = self._create_session()

        # Base URL
        self.base_url = self.provider_config.get('base_url', '')

        # API key (if applicable)
        self.api_key = self.provider_config.get('api_key', '')

    def _create_session(self) -> requests.Session:
        """Create requests session with retry logic."""
        session = requests.Session()

        # Configure retries
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=self.backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # Set timeout
        timeout = self.provider_config.get('timeout', 30)
        session.timeout = timeout

        return session

    def _check_rate_limits(self):
        """Check and acquire tokens from all rate limiters."""
        for limiter_name, limiter in self.rate_limiters.items():
            if not limiter.acquire(blocking=True):
                raise RateLimitError(
                    f"Rate limit exceeded for {limiter_name} on {self.config_key}"
                )

    def _cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        # Create a deterministic string from args and kwargs
        key_data = {
            'args': args,
            'kwargs': sorted(kwargs.items())
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Get data from cache if not expired."""
        if cache_key in self.cache:
            data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                self.logger.debug(f"Cache hit for {cache_key}")
                return data
            else:
                # Cache expired
                del self.cache[cache_key]

        return None

    def _save_to_cache(self, cache_key: str, data: Any):
        """Save data to cache with timestamp."""
        self.cache[cache_key] = (data, time.time())

    def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        method: str = "GET",
        **kwargs
    ) -> requests.Response:
        """
        Make HTTP request with rate limiting and retry logic.

        Args:
            endpoint: API endpoint
            params: Query parameters
            method: HTTP method
            **kwargs: Additional requests arguments

        Returns:
            Response object

        Raises:
            DataError: If request fails after retries
        """
        # Check rate limits
        self._check_rate_limits()

        # Build URL
        url = f"{self.base_url}{endpoint}"

        # Make request
        try:
            if method.upper() == "GET":
                response = self.session.get(url, params=params, **kwargs)
            elif method.upper() == "POST":
                response = self.session.post(url, json=params, **kwargs)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            response.raise_for_status()
            return response

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                raise RateLimitError(f"Rate limit exceeded: {e}")
            else:
                raise DataError(f"HTTP error: {e}")

        except requests.exceptions.RequestException as e:
            raise DataError(f"Request failed: {e}")

    @abstractmethod
    def get_daily_prices(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get daily OHLCV data.

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with OHLCV data
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available and configured."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config_key='{self.config_key}')"
