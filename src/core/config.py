"""
Configuration management for QuantCLI.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class ConfigManager:
    """
    Centralized configuration management.

    Loads configuration from YAML files and environment variables.
    """

    _instance = None
    _configs: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize configuration by loading env vars and config files."""
        # Load environment variables
        env_path = Path(__file__).parent.parent.parent / '.env'
        if env_path.exists():
            load_dotenv(env_path)

        # Load config files
        config_dir = Path(__file__).parent.parent.parent / 'config'
        config_files = [
            'data_sources.yaml',
            'models.yaml',
            'risk.yaml',
            'backtest.yaml'
        ]

        for config_file in config_files:
            config_path = config_dir / config_file
            if config_path.exists():
                config_name = config_file.replace('.yaml', '')
                self._configs[config_name] = self._load_yaml(config_path)

    def _load_yaml(self, filepath: Path) -> Dict[str, Any]:
        """Load YAML configuration file with environment variable substitution."""
        with open(filepath, 'r') as f:
            content = f.read()

        # Replace environment variables
        import re
        pattern = r'\$\{([^}]+)\}'

        def replacer(match):
            var_name = match.group(1)
            return os.getenv(var_name, match.group(0))

        content = re.sub(pattern, replacer, content)
        return yaml.safe_load(content)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key: Configuration key (e.g., 'data_sources.alpha_vantage.api_key')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        parts = key.split('.')
        value = self._configs

        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default

        return value

    def get_config(self, config_name: str) -> Dict[str, Any]:
        """
        Get entire configuration dictionary.

        Args:
            config_name: Name of config file (without .yaml extension)

        Returns:
            Configuration dictionary
        """
        return self._configs.get(config_name, {})

    def reload(self):
        """Reload all configuration files."""
        self._configs.clear()
        self._initialize()


class DatabaseSettings(BaseSettings):
    """Database configuration."""
    database_url: str = Field(default="postgresql://quantcli:changeme@localhost:5432/quantcli")
    pool_size: int = 20
    max_overflow: int = 40
    pool_timeout: int = 30
    pool_recycle: int = 3600

    class Config:
        env_file = ".env"


class RedisSettings(BaseSettings):
    """Redis configuration."""
    redis_nodes: str = Field(default="localhost:7000,localhost:7001,localhost:7002")
    redis_password: str = Field(default="")

    @property
    def nodes_list(self):
        """Parse Redis nodes string into list of tuples."""
        nodes = []
        for node in self.redis_nodes.split(','):
            host, port = node.strip().split(':')
            nodes.append((host, int(port)))
        return nodes

    class Config:
        env_file = ".env"


class KafkaSettings(BaseSettings):
    """Kafka configuration."""
    kafka_bootstrap_servers: str = Field(default="localhost:29092")
    schema_registry_url: str = Field(default="http://localhost:8081")

    class Config:
        env_file = ".env"


class IBKRSettings(BaseSettings):
    """Interactive Brokers configuration."""
    ibkr_account: str = Field(default="DU1234567")
    ibkr_host: str = Field(default="127.0.0.1")
    ibkr_port: int = Field(default=7497)
    ibkr_client_id: int = Field(default=1)

    class Config:
        env_file = ".env"


class MLFlowSettings(BaseSettings):
    """MLFlow configuration."""
    mlflow_tracking_uri: str = Field(default="http://localhost:5000")
    mlflow_artifact_root: str = Field(default="/mlflow/artifacts")

    class Config:
        env_file = ".env"


class ObservabilitySettings(BaseSettings):
    """Observability configuration."""
    prometheus_url: str = Field(default="http://localhost:9090")
    grafana_url: str = Field(default="http://localhost:3000")
    jaeger_url: str = Field(default="http://localhost:16686")
    otel_exporter_otlp_endpoint: str = Field(default="http://localhost:4317")

    class Config:
        env_file = ".env"


class RiskSettings(BaseSettings):
    """Risk management configuration."""
    max_portfolio_value_usd: float = Field(default=250000.0)
    max_daily_loss_pct: float = Field(default=2.0)
    max_drawdown_pct: float = Field(default=15.0)
    max_position_size_pct: float = Field(default=2.0)
    trading_mode: str = Field(default="paper")
    enable_kill_switch: bool = Field(default=True)
    pre_trade_checks_enabled: bool = Field(default=True)

    class Config:
        env_file = ".env"


# Global config instance
config = ConfigManager()
