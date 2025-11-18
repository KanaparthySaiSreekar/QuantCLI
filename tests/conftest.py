"""
Pytest configuration and shared fixtures for QuantCLI tests.
"""

import os
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock, MagicMock


# Set test environment
os.environ['QUANTCLI_ENV'] = 'test'


@pytest.fixture(scope="session")
def test_data_dir():
    """Return path to test data directory."""
    return Path(__file__).parent / 'fixtures'


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV market data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')

    # Generate realistic price data
    base_price = 100.0
    returns = np.random.normal(0.0005, 0.02, len(dates))
    prices = base_price * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.uniform(-0.01, 0.01, len(dates))),
        'high': prices * (1 + np.random.uniform(0.0, 0.02, len(dates))),
        'low': prices * (1 + np.random.uniform(-0.02, 0.0, len(dates))),
        'close': prices,
        'volume': np.random.randint(1_000_000, 10_000_000, len(dates))
    })

    df.set_index('timestamp', inplace=True)
    return df


@pytest.fixture
def sample_symbol_list():
    """Return list of test stock symbols."""
    return ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']


@pytest.fixture
def mock_config():
    """Mock ConfigManager for testing."""
    config = MagicMock()
    config.get.return_value = {
        'enabled': True,
        'api_key': 'test_api_key_1234567890abcdef',
        'base_url': 'https://api.example.com',
        'rate_limits': {
            'calls_per_day': 500,
            'calls_per_minute': 5
        },
        'retry': {
            'max_attempts': 3,
            'backoff_factor': 2,
            'backoff_max': 60
        },
        'cache_ttl_seconds': 3600,
        'timeout': 30
    }
    return config


@pytest.fixture
def mock_api_response():
    """Mock successful API response."""
    response = Mock()
    response.status_code = 200
    response.json.return_value = {
        'status': 'success',
        'data': {'price': 150.25, 'volume': 1000000}
    }
    response.text = '{"status": "success"}'
    return response


@pytest.fixture
def mock_failed_api_response():
    """Mock failed API response."""
    response = Mock()
    response.status_code = 500
    response.json.return_value = {'error': 'Internal server error'}
    response.text = '{"error": "Internal server error"}'
    response.raise_for_status.side_effect = Exception("500 Server Error")
    return response


@pytest.fixture
def sample_model_predictions():
    """Sample model prediction dictionary."""
    return {
        'direction': 1,  # Buy signal
        'confidence': 0.75,
        'probability': 0.65,
        'features_used': 25
    }


@pytest.fixture
def sample_features():
    """Sample feature dataframe."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    return pd.DataFrame({
        'rsi_14': np.random.uniform(30, 70, len(dates)),
        'macd': np.random.uniform(-2, 2, len(dates)),
        'bb_upper': np.random.uniform(105, 110, len(dates)),
        'bb_lower': np.random.uniform(90, 95, len(dates)),
        'volume_sma_20': np.random.randint(5_000_000, 15_000_000, len(dates))
    }, index=dates)


@pytest.fixture
def temp_config_file(tmp_path):
    """Create temporary config file for testing."""
    config_content = """
alpha_vantage:
  enabled: true
  api_key: ${ALPHA_VANTAGE_API_KEY}
  rate_limits:
    calls_per_day: 25
    calls_per_minute: 5
"""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(config_content)
    return config_file


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances between tests to avoid state pollution."""
    from src.core.config import ConfigManager

    # Reset ConfigManager singleton
    ConfigManager._instance = None

    yield

    # Cleanup after test
    ConfigManager._instance = None


@pytest.fixture
def mock_database_connection():
    """Mock database connection."""
    conn = MagicMock()
    conn.execute.return_value = None
    conn.commit.return_value = None
    conn.rollback.return_value = None
    conn.close.return_value = None
    return conn


@pytest.fixture
def sample_signal():
    """Sample trading signal for testing."""
    from src.signals.generator import Signal, SignalType

    return Signal(
        symbol='AAPL',
        timestamp=datetime.now(),
        signal_type=SignalType.BUY,
        strength=0.75,
        confidence=0.80,
        metadata={'test': True}
    )


@pytest.fixture
def mock_logger():
    """Mock logger instance."""
    logger = MagicMock()
    logger.debug = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.critical = Mock()
    return logger


# Marker for tests that require external services
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "requires_api: mark test as requiring external API"
    )
    config.addinivalue_line(
        "markers", "requires_db: mark test as requiring database"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
