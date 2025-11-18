"""
Integration tests for DataOrchestrator.

Tests provider failover, caching, and data quality validation.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.data.orchestrator import DataOrchestrator
from src.core.exceptions import DataError


class TestDataOrchestrator:
    """Integration tests for DataOrchestrator."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration."""
        config = MagicMock()
        config.get.return_value = {
            'alpha_vantage': {
                'enabled': True,
                'priority': 1
            },
            'finnhub': {
                'enabled': True,
                'priority': 2
            }
        }
        return config

    @pytest.fixture
    def orchestrator(self, mock_config):
        """Create orchestrator with mocked config."""
        with patch('src.data.orchestrator.ConfigManager', return_value=mock_config):
            orch = DataOrchestrator(mock_config)
            return orch

    def test_init_providers(self, orchestrator):
        """Test provider initialization."""
        assert hasattr(orchestrator, 'providers')
        assert hasattr(orchestrator, 'provider_priority')

    def test_provider_failover(self, orchestrator, sample_ohlcv_data):
        """Test failover when primary provider fails."""
        # Mock providers
        mock_provider1 = Mock()
        mock_provider1.get_daily_prices.side_effect = DataError("API error")

        mock_provider2 = Mock()
        mock_provider2.get_daily_prices.return_value = sample_ohlcv_data

        orchestrator.providers = {
            'provider1': mock_provider1,
            'provider2': mock_provider2
        }
        orchestrator.provider_priority = ['provider1', 'provider2']

        # Should fallback to provider2
        result = orchestrator.get_daily_data('AAPL')

        assert not result.empty
        assert len(result) == len(sample_ohlcv_data)
        mock_provider1.get_daily_prices.assert_called_once()
        mock_provider2.get_daily_prices.assert_called_once()

    def test_all_providers_fail(self, orchestrator):
        """Test when all providers fail."""
        mock_provider = Mock()
        mock_provider.get_daily_prices.side_effect = DataError("API error")

        orchestrator.providers = {'provider1': mock_provider}
        orchestrator.provider_priority = ['provider1']

        with pytest.raises(DataError, match="All data providers failed"):
            orchestrator.get_daily_data('AAPL')

    def test_data_reconciliation(self, orchestrator, sample_ohlcv_data):
        """Test data reconciliation from multiple providers."""
        # Create slightly different data from each provider
        data1 = sample_ohlcv_data.copy()
        data2 = sample_ohlcv_data.copy()
        data2['close'] = data2['close'] * 1.001  # 0.1% difference

        mock_provider1 = Mock()
        mock_provider1.get_daily_prices.return_value = data1

        mock_provider2 = Mock()
        mock_provider2.get_daily_prices.return_value = data2

        orchestrator.providers = {
            'provider1': mock_provider1,
            'provider2': mock_provider2
        }

        # Should use median of values
        result = orchestrator.reconcile_data('AAPL')

        assert not result.empty
        # Median should be between the two values
        assert (result['close'].iloc[0] >= min(data1['close'].iloc[0], data2['close'].iloc[0]))
        assert (result['close'].iloc[0] <= max(data1['close'].iloc[0], data2['close'].iloc[0]))

    @pytest.mark.slow
    def test_batch_data_fetch(self, orchestrator, sample_ohlcv_data, sample_symbol_list):
        """Test batch data fetching for multiple symbols."""
        mock_provider = Mock()
        mock_provider.get_daily_prices.return_value = sample_ohlcv_data

        orchestrator.providers = {'provider1': mock_provider}
        orchestrator.provider_priority = ['provider1']

        results = orchestrator.get_batch_data(sample_symbol_list[:3])

        assert len(results) == 3
        assert all(symbol in results for symbol in sample_symbol_list[:3])
        assert all(not df.empty for df in results.values())
