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
cp config/env.example .env
# Edit .env with your keys

# 4. Initialize database
python scripts/init_database.py

# 5. Download ML models
python scripts/download_models.py

# 6. Run backtest
python scripts/run_backtest.py --strategy momentum --validation cpcv

# 7. Start live trading (paper account)
python scripts/start_trading.py --mode paper --broker ibkr
```

## Performance Expectations

**Conservative (simple factors)**: 8-12% annual return, Sharpe 0.8-1.2, 15-25% max drawdown

**Sophisticated (ensemble+GPU)**: 12-18% annual return, Sharpe 1.2-1.8, 12-20% max drawdown

*(Reference: Berkshire Hathaway Sharpe 0.79 from 1976-2017)*

## Documentation

Full documentation in `/docs`:
- [Installation Guide](docs/installation.md)
- [Configuration Reference](docs/configuration.md)
- [API Documentation](docs/api.md)
- [Strategy Development](docs/strategies.md)
- [Compliance Guide](docs/compliance.md)

## License

Apache License 2.0

## Disclaimer

**Educational purposes only. Trading involves substantial risk of loss.**