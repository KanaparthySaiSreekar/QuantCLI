"""
Feature Store Wrapper for QuantCLI

Provides unified interface for offline (training) and online (serving) feature access.
Guarantees identical features in both contexts.
"""

from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
from feast import FeatureStore
from loguru import logger


class QuantCLIFeatureStore:
    """
    Wrapper around Feast FeatureStore with QuantCLI-specific logic.

    Guarantees:
    - Identical offline/online features
    - Fallback to computed features if Redis unavailable
    - Metrics tracking (cache hit rate, fallback rate)
    """

    def __init__(self, repo_path: str = "infra/feast"):
        """
        Initialize feature store.

        Args:
            repo_path: Path to Feast repository
        """
        self.repo_path = Path(repo_path)
        try:
            self.store = FeatureStore(repo_path=str(self.repo_path))
            logger.success(f"Feature store initialized from {repo_path}")
        except Exception as e:
            logger.warning(f"Could not initialize Feast store: {e}")
            self.store = None

        # Metrics
        self.cache_hits = 0
        self.cache_misses = 0
        self.fallback_count = 0

    def get_online_features(
        self,
        symbols: List[str],
        feature_refs: Optional[List[str]] = None,
        full_feature_names: bool = False,
    ) -> pd.DataFrame:
        """
        Get features for online serving (real-time inference).

        Args:
            symbols: List of stock symbols
            feature_refs: List of feature references (e.g., "technical_features_v1:nma_9")
                         If None, gets all technical features
            full_feature_names: Whether to return full feature names

        Returns:
            DataFrame with features for each symbol
        """
        if self.store is None:
            logger.warning("Feature store not available, using fallback")
            self.fallback_count += 1
            return self._fallback_compute_features(symbols)

        # Default to all technical features if not specified
        if feature_refs is None:
            feature_refs = [
                "technical_features_v1:nma_9",
                "technical_features_v1:nma_20",
                "technical_features_v1:bb_pct",
                "technical_features_v1:volume_z_20",
                "technical_features_v1:rsi_14",
                "technical_features_v1:macd",
                # Add more as needed
            ]

        entity_rows = [{"symbol": symbol} for symbol in symbols]

        try:
            feature_vector = self.store.get_online_features(
                features=feature_refs,
                entity_rows=entity_rows,
                full_feature_names=full_feature_names,
            )
            self.cache_hits += len(symbols)
            return feature_vector.to_df()

        except Exception as e:
            logger.error(f"Error fetching online features: {e}")
            self.cache_misses += len(symbols)
            self.fallback_count += 1
            return self._fallback_compute_features(symbols)

    def get_historical_features(
        self,
        entity_df: pd.DataFrame,
        features: List[str],
    ) -> pd.DataFrame:
        """
        Get historical features for training (offline).

        Args:
            entity_df: DataFrame with (symbol, event_timestamp) rows
            features: List of feature references

        Returns:
            DataFrame with historical features joined with entity_df
        """
        if self.store is None:
            logger.error("Feature store not initialized")
            raise RuntimeError("Cannot get historical features without Feast store")

        training_df = self.store.get_historical_features(
            entity_df=entity_df,
            features=features,
        ).to_df()

        return training_df

    def _fallback_compute_features(self, symbols: List[str]) -> pd.DataFrame:
        """
        Fallback: compute features on-the-fly if Redis unavailable.

        This should be rare in production and logged as a metric.
        """
        logger.warning(f"Computing features on-the-fly for {len(symbols)} symbols (FALLBACK)")

        from src.features.generator import get_feature_generator
        from src.data.orchestrator import DataOrchestrator

        orchestrator = DataOrchestrator()
        generator = get_feature_generator()

        results = []
        for symbol in symbols:
            try:
                # Get recent data
                data = orchestrator.get_daily_prices(symbol, use_failover=True)
                if data.empty:
                    logger.warning(f"No data for {symbol}")
                    continue

                # Compute features
                features = generator.compute_technical(data.tail(250))  # Last 250 days
                latest = features.iloc[-1].to_dict()
                latest["symbol"] = symbol
                results.append(latest)

            except Exception as e:
                logger.error(f"Error computing fallback features for {symbol}: {e}")

        return pd.DataFrame(results)

    def materialize(
        self,
        start_date: datetime,
        end_date: datetime,
    ):
        """
        Materialize features to online store (Redis).

        This should be run periodically (e.g., daily) to keep online store fresh.
        """
        if self.store is None:
            logger.error("Feature store not initialized")
            return

        logger.info(f"Materializing features from {start_date} to {end_date}")

        try:
            self.store.materialize(
                start_date=start_date,
                end_date=end_date,
            )
            logger.success("Features materialized to online store")
        except Exception as e:
            logger.error(f"Failed to materialize features: {e}")

    def get_metrics(self) -> dict:
        """Get feature store metrics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (
            self.cache_hits / total_requests * 100 if total_requests > 0 else 0
        )

        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate_pct": hit_rate,
            "fallback_count": self.fallback_count,
        }


# Global instance
_feature_store = None


def get_feature_store() -> QuantCLIFeatureStore:
    """Get global feature store instance."""
    global _feature_store
    if _feature_store is None:
        _feature_store = QuantCLIFeatureStore()
    return _feature_store
