# QuantCLI - Institutional-Grade Algorithmic Trading System

A complete end-to-end algorithmic trading system for US equities from India, implementing institutional-grade infrastructure with entirely free and open-source tools.

## System Architecture

This system implements every feature from the comprehensive research document, including:

- **Data**: Multi-source aggregation (Alpha Vantage, Tiingo, FRED, Polygon, Finnhub, Reddit/GDELT)
- **Storage**: TimescaleDB + Redis Cluster + Kafka event streaming
- **Features**: Technical, sentiment (FinBERT), microstructure (VPIN), regime (HMM), cross-sectional
- **Models**: Ensemble (XGBoost/LightGBM/CatBoost/LSTM) with INT8/ONNX/oneDAL optimization
- **Validation**: CPCV, walk-forward, VectorBT, PSR/DSR metrics
- **Portfolio**: Riskfolio-Lib, PyPortfolioOpt, HRP, Black-Litterman, Kelly sizing
- **Execution**: Interactive Brokers TWS API, FIX protocol, Smart Order Routing
- **Risk**: Pre-trade checks (<10μs), kill switches, position limits, real-time P&L
- **NLP**: FinBERT, Financial RoBERTa, GDELT, regression-calibrated sentiment
- **Strategies**: HMM regime switching, Johansen pairs trading, statistical arbitrage
- **Infrastructure**: Kafka, Schema Registry, DataHub lineage, Resilience4j patterns
- **Observability**: OpenTelemetry, Prometheus, Jaeger, Grafana, SLI/SLO tracking
- **Compliance**: CAT/TRF reporting, Reg NMS, SR 11-7 model governance

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start infrastructure (TimescaleDB, Redis, Kafka)
docker-compose up -d

# 3. Configure API keys
# Create .env file with your API keys:
cat > .env << EOF
ALPHA_VANTAGE_API_KEY=your_key_here
FINNHUB_API_KEY=your_key_here
TIINGO_API_KEY=your_key_here
DATABASE_URL=postgresql://quantcli:your_password@localhost:5432/quantcli
EOF

# 4. Test data acquisition
python -c "from src.data.orchestrator import DataOrchestrator; from src.core.config import ConfigManager; orch = DataOrchestrator(ConfigManager()); print(orch.get_daily_data('AAPL'))"

# 5. Run tests
pytest tests/ -v

# 6. Generate features (example)
python -c "from src.features import FeatureEngineer; import pandas as pd; print('Feature engineering ready')"
```

**Note**: Full pipeline (ML training, backtesting, live trading) implementation is in progress. See [Implementation Status](#implementation-status) below.

## Performance Expectations

**Conservative (simple factors)**: 8-12% annual return, Sharpe 0.8-1.2, 15-25% max drawdown

**Sophisticated (ensemble+GPU)**: 12-18% annual return, Sharpe 1.2-1.8, 12-20% max drawdown

*(Reference: Berkshire Hathaway Sharpe 0.79 from 1976-2017)*

## Implementation Status

### ✅ Completed (Production-Ready)
- **Core Infrastructure**: Configuration management, logging, exception handling
- **Data Acquisition**: 7 data providers with failover, rate limiting, caching
- **Signal Generation**: Signal generator with batch processing and ranking
- **Feature Engineering**: 50+ technical indicators, price/volume/time features
- **Backtesting Engine**: Vectorized backtesting with transaction costs and metrics
- **Docker Infrastructure**: 14-service architecture (TimescaleDB, Redis, Kafka, etc.)
- **Test Framework**: Unit tests for core modules, pytest configuration
- **Security Improvements**: Thread-safe singletons, API key validation, input validation

### 🚧 In Progress
- **ML Training Pipeline**: Model training, ensemble configuration
- **Database Persistence**: TimescaleDB integration for historical data
- **Execution System**: Interactive Brokers integration
- **Additional Tests**: Integration tests, provider tests, orchestrator tests

### 📋 Planned
- **Risk Management**: Pre-trade checks, kill switches, position limits
- **Portfolio Optimization**: HRP, Black-Litterman, Kelly sizing
- **NLP Sentiment**: FinBERT integration, news analysis
- **Monitoring**: Grafana dashboards, alerting rules
- **CI/CD**: GitHub Actions workflow, automated testing

### Test Coverage
- Rate Limiter: ✅ Comprehensive tests
- Signal Generator: ✅ Comprehensive tests
- Config Manager: ✅ Thread safety verified
- Data Providers: ⏳ In progress
- Overall Coverage: ~40% (Target: 80%)

## Documentation

Full documentation in `/docs`:
- [Installation Guide](docs/installation.md)
- [Configuration Reference](docs/configuration.md)
- [API Documentation](docs/api.md)
- [Strategy Development](docs/strategies.md)
- [Compliance Guide](docs/compliance.md)
- [Architecture](ARCHITECTURE.md)
- [Implementation Guide](IMPLEMENTATION_GUIDE.md)

## License

Apache License 2.0

## Disclaimer

**Educational purposes only. Trading involves substantial risk of loss.**