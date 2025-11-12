# QuantCLI System Architecture

## Overview

QuantCLI is built as a modular, event-driven system with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Application Layer                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   Trading    │  │  Backtesting │  │  Model Training      │  │
│  │   Engine     │  │   Engine     │  │  Pipeline            │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        Business Logic Layer                      │
│  ┌──────────┐ ┌───────────┐ ┌──────────┐ ┌──────────────────┐  │
│  │ Strategy │ │ Portfolio │ │   Risk   │ │   Execution      │  │
│  │ Manager  │ │ Optimizer │ │  Manager │ │   Router         │  │
│  └──────────┘ └───────────┘ └──────────┘ └──────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                         Service Layer                            │
│  ┌─────────┐ ┌──────────┐ ┌────────────┐ ┌──────────────────┐  │
│  │  Data   │ │ Feature  │ │   Model    │ │  Observability   │  │
│  │Providers│ │ Engineer │ │  Service   │ │   Service        │  │
│  └─────────┘ └──────────┘ └────────────┘ └──────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Infrastructure Layer                        │
│  ┌────────────┐ ┌────────┐ ┌────────┐ ┌───────────────────┐    │
│  │TimescaleDB │ │ Redis  │ │ Kafka  │ │ Prometheus/Jaeger │    │
│  │            │ │Cluster │ │        │ │                   │    │
│  └────────────┘ └────────┘ └────────┘ └───────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

### Market Data Pipeline
```
External APIs → Rate Limiter → Cache → Validation → TimescaleDB
     ↓              ↓            ↓          ↓            ↓
  (Alpha V)    (Token Bucket)  (Redis)  (Quality)  (Hypertable)
  (Tiingo)
  (FRED)
```

### Feature Engineering Pipeline
```
Raw Data → Technical Features → Sentiment Features → Microstructure Features
              ↓                      ↓                        ↓
         (Normalized MA)        (FinBERT)                (VPIN)
         (Bollinger)            (Calibration)            (OFI)
              ↓                      ↓                        ↓
                    ┌────────────────────────────┐
                    │    Feature Store (Redis)    │
                    └────────────────────────────┘
```

### Model Training Pipeline
```
Features → Train/Test Split → Ensemble Training → Optimization → Deployment
                ↓                    ↓                 ↓             ↓
             (CPCV)            (XGB/LGBM/CB)       (INT8)       (Production)
                               (Stacking)          (ONNX)
```

### Execution Pipeline
```
Signal → Risk Check → Order → Router → Venue → Fill Confirmation
           ↓            ↓        ↓       ↓           ↓
       (<10μs)     (Position) (IBKR) (Smart)    (Kafka Event)
                   (Limits)           (IOC/ISO)
```

## Component Details

### src/data/
- **providers/**: Individual data source implementations
- **storage/**: Database access layer
- **preprocessing/**: Data cleaning and validation
- **orchestrator.py**: Multi-source coordination with failover

### src/features/
- **technical/**: Price-based indicators
- **sentiment/**: NLP sentiment analysis
- **microstructure/**: Order flow and market structure
- **regime/**: HMM-based market regime detection
- **cross_sectional/**: Relative value features

### src/models/
- **ensemble/**: Stacking, blending, voting
- **lstm/**: Sequence models for time series
- **optimization/**: Quantization, ONNX, distillation
- **distillation/**: Knowledge transfer from ensemble to single model

### src/backtesting/
- **validation/**: CPCV, walk-forward, time-series splits
- **metrics/**: Sharpe, PSR, DSR, drawdown analysis
- **costs/**: Transaction cost and slippage modeling

### src/portfolio/
- **optimization/**: Riskfolio-Lib, PyPortfolioOpt integration
- **sizing/**: Kelly, volatility-based, risk parity
- **rebalancing/**: Periodic and threshold-based rebalancing

### src/execution/
- **ibkr/**: Interactive Brokers TWS API client
- **fix/**: FIX protocol implementation
- **routing/**: Smart order routing across venues
- **venues/**: Venue-specific adapters

### src/risk/
- **checks/**: Pre-trade validation (<10μs target)
- **limits/**: Position, order, loss limits
- **monitoring/**: Real-time P&L and exposure tracking
- **pnl/**: Position and portfolio P&L calculation

### src/nlp/
- **models/**: FinBERT, Financial RoBERTa
- **sentiment/**: Sentiment scoring and calibration
- **ner/**: Named entity recognition for tickers
- **news/**: News aggregation and parsing

### src/strategies/
- **regime/**: Regime-switching strategies
- **pairs/**: Statistical arbitrage and pairs trading
- **momentum/**: Time-series and cross-sectional momentum
- **arbitrage/**: Mean reversion and convergence trades
- **factors/**: Factor-based strategies

### src/infrastructure/
- **cache/**: Redis cluster management
- **messaging/**: Kafka producers/consumers
- **database/**: TimescaleDB connection pooling
- **lineage/**: DataHub integration for data lineage

### src/observability/
- **metrics/**: Prometheus metrics collection
- **tracing/**: OpenTelemetry distributed tracing
- **alerting/**: Alert management and routing
- **dashboards/**: Grafana dashboard definitions

### src/compliance/
- **reporting/**: CAT, TRF reporting
- **governance/**: SR 11-7 model governance
- **audit/**: Audit trail and compliance checks

## Configuration Management

All components use centralized configuration:

```python
from src.core.config import ConfigManager

config = ConfigManager()
api_key = config.get('data_sources.alpha_vantage.api_key')
risk_limits = config.get_config('risk')
```

Configuration hierarchy:
1. Environment variables (.env)
2. YAML config files (config/*.yaml)
3. Default values in code

## Error Handling

Standardized exception hierarchy:

```
QuantCLIError (base)
├── DataError
│   └── RateLimitError
├── ModelError
├── ExecutionError
├── RiskError
│   ├── KillSwitchError
│   ├── PositionLimitError
│   └── PreTradeCheckError
└── ValidationError
```

## Logging

Structured logging via loguru:

```python
from src.core.logging_config import get_logger

logger = get_logger(__name__)
logger.info(f"Processing {symbol}", symbol=symbol, timestamp=datetime.now())
```

Logs include:
- Timestamp (millisecond precision)
- Level
- Module/function/line
- Structured context

## Deployment

### Development
```bash
docker-compose up -d
python scripts/start_trading.py --mode paper
```

### Production
```bash
# Start infrastructure
docker-compose -f docker-compose.prod.yml up -d

# Deploy application
kubectl apply -f k8s/

# Monitor
kubectl logs -f deployment/quantcli-trading-engine
```

## Performance Targets

| Component | Target Latency | Achieved |
|-----------|---------------|----------|
| Pre-trade checks | <10μs | TBD |
| Feature calculation | <100ms | TBD |
| Model inference (CatBoost INT8) | <1ms | TBD |
| Order submission | <50ms | TBD |
| End-to-end signal→order | <200ms | TBD |

## Scalability

- **Horizontal**: Stateless services scale with Kubernetes
- **Vertical**: TimescaleDB scales to 10M+ ticks/day
- **Caching**: Redis Cluster handles 100K+ ops/sec
- **Messaging**: Kafka handles 1M+ events/sec

## Security

- API keys stored in environment variables
- Database credentials rotated monthly
- TLS for all external connections
- Pre-trade checks prevent fat-finger errors
- Kill switches prevent runaway losses

---

This architecture supports institutional-grade trading while using entirely free and open-source tools.
