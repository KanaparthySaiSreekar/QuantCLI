"""
Data acquisition and storage module.
"""

from .orchestrator import DataOrchestrator
from .providers.alpha_vantage import AlphaVantageProvider
from .providers.tiingo import TiingoProvider
from .providers.fred import FREDProvider
from .providers.finnhub import FinnhubProvider
from .providers.polygon import PolygonProvider
from .providers.reddit import RedditProvider
from .providers.gdelt import GDELTProvider

__all__ = [
    'DataOrchestrator',
    'AlphaVantageProvider',
    'TiingoProvider',
    'FREDProvider',
    'FinnhubProvider',
    'PolygonProvider',
    'RedditProvider',
    'GDELTProvider'
]
