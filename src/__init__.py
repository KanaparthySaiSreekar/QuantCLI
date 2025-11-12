"""
QuantCLI - Institutional-Grade Algorithmic Trading System

A complete end-to-end algorithmic trading system for US equities from India.
"""

__version__ = "1.0.0"
__author__ = "QuantCLI Team"
__license__ = "Apache 2.0"

from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DATA_DIR = DATA_DIR / "features"
MODELS_DIR = DATA_DIR / "models"

# Config directory
CONFIG_DIR = PROJECT_ROOT / "config"

# Logs directory
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
                  FEATURES_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
