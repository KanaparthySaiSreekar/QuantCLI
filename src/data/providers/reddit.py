"""
Reddit data provider for sentiment analysis.
100 queries/minute.
"""

from datetime import datetime
from typing import Optional, List
import pandas as pd
import praw

from .base import BaseDataProvider
from src.core.exceptions import DataError


class RedditProvider(BaseDataProvider):
    """Reddit provider for social sentiment."""

    def __init__(self):
        super().__init__('reddit')
        
        client_id = self.provider_config.get('client_id')
        client_secret = self.provider_config.get('client_secret')
        user_agent = self.provider_config.get('user_agent')
        
        if not all([client_id, client_secret, user_agent]):
            raise DataError("Reddit credentials not configured")
        
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )

    def is_available(self) -> bool:
        return self.provider_config.get('enabled', False)

    def get_daily_prices(self, symbol: str, start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None) -> pd.DataFrame:
        """Not applicable for Reddit."""
        raise NotImplementedError("Use get_posts() or get_sentiment()")

    def get_posts(self, subreddit: str = 'wallstreetbets',
                  limit: int = 100, time_filter: str = 'day') -> pd.DataFrame:
        """Get posts from subreddit."""
        cache_key = self._cache_key('posts', subreddit, limit, time_filter)
        
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data

        try:
            sub = self.reddit.subreddit(subreddit)
            posts = []
            
            for post in sub.top(time_filter=time_filter, limit=limit):
                posts.append({
                    'id': post.id,
                    'title': post.title,
                    'score': post.score,
                    'num_comments': post.num_comments,
                    'created_utc': datetime.fromtimestamp(post.created_utc),
                    'selftext': post.selftext,
                    'url': post.url,
                    'upvote_ratio': post.upvote_ratio
                })
            
            df = pd.DataFrame(posts)
            if not df.empty:
                df = df.set_index('created_utc').sort_index()
            
            self._save_to_cache(cache_key, df)
            return df

        except Exception as e:
            raise DataError(f"Reddit error: {e}")
