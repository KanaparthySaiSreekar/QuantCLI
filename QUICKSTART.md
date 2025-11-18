# QuantCLI Quick Start Guide

Get your algorithmic trading system running locally in 5 minutes.

## Prerequisites

- Docker Desktop (with Docker Compose)
- Python 3.10+
- 8GB RAM minimum (16GB recommended)
- 50GB free disk space

## Step 1: Clone and Setup (1 minute)

```bash
# Clone repository
git clone https://github.com/KanaparthySaiSreekar/QuantCLI.git
cd QuantCLI

# Install Python dependencies
pip install -r requirements.txt
```

## Step 2: Configure API Keys (2 minutes)

```bash
# Copy environment template
cp config/env.example .env

# Edit .env and add your API keys
nano .env
```

**Required API Keys (all free tier):**
- **Alpha Vantage**: https://www.alphavantage.co/support/#api-key
- **Tiingo**: https://www.tiingo.com/account/api/token
- **FRED**: https://fred.stlouisfed.org/docs/api/api_key.html

**Optional API Keys:**
- Finnhub: https://finnhub.io/register
- Polygon: https://polygon.io/

## Step 3: Start Infrastructure (1 minute)

```bash
# Start all services
docker-compose -f docker-compose.local.yml up -d

# Wait for services to be ready (~30 seconds)
docker-compose -f docker-compose.local.yml ps
```

**Services Started:**
- ✅ TimescaleDB → `localhost:5432`
- ✅ Redis → `localhost:6379`
- ✅ Kafka → `localhost:9092`
- ✅ Prometheus → `localhost:9090`
- ✅ Grafana → `localhost:3000` (admin/admin)
- ✅ MLflow → `localhost:5000`
- ✅ Jaeger → `localhost:16686`

## Step 4: Initialize Database (30 seconds)

```bash
python scripts/init_database.py
```

This creates all database tables and hypertables optimized for time-series data.

## Step 5: Download Market Data (2-5 minutes)

```bash
# Download 1 year of data for default symbols
python scripts/update_data.py

# Or specify custom symbols
python scripts/update_data.py --symbols AAPL,MSFT,GOOGL,TSLA,NVDA
```

## Step 6: Train Models (Optional, 5-10 minutes)

```bash
# Train ensemble models
python scripts/train_ensemble.py

# Or with optimization
python scripts/train_ensemble.py --optimize
```

## Step 7: Start Paper Trading 🚀

```bash
# Start paper trading with $100,000 virtual capital
python scripts/start_trading.py --mode paper --capital 100000 --symbols AAPL,MSFT,GOOGL,TSLA,NVDA
```

**You're now trading! (Paper mode - no real money)**

---

## Quick Commands Reference

### Daily Operations

```bash
# Update market data (run daily)
python scripts/update_data.py

# Check data statistics
python scripts/update_data.py --stats

# Start paper trading
python scripts/start_trading.py --mode paper

# Run backtest
python scripts/run_backtest.py --symbols AAPL,MSFT --start 2023-01-01

# Retrain models (run weekly)
python scripts/train_ensemble.py
```

### Monitoring

```bash
# View Grafana dashboards
open http://localhost:3000
# Login: admin/admin

# View MLflow experiments
open http://localhost:5000

# View Jaeger traces
open http://localhost:16686

# View Prometheus metrics
open http://localhost:9090
```

### Docker Management

```bash
# Check services status
docker-compose -f docker-compose.local.yml ps

# View logs
docker-compose -f docker-compose.local.yml logs -f

# Stop all services
docker-compose -f docker-compose.local.yml down

# Stop and remove volumes (clean slate)
docker-compose -f docker-compose.local.yml down -v
```

---

## What's Running?

After completing the quick start, you have:

✅ **Infrastructure**
- TimescaleDB with market data hypertables
- Redis cache for features
- Kafka for event streaming
- Prometheus + Grafana for monitoring

✅ **Data Pipeline**
- Multi-source data acquisition with failover
- Automatic rate limiting and caching
- Time-series optimized storage

✅ **Trading System**
- Paper trading with realistic simulation
- Signal generation (<200ms latency)
- Pre-trade risk checks
- Real-time P&L tracking

✅ **Monitoring**
- Trading performance dashboards
- System health metrics
- Distributed tracing
- ML experiment tracking

---

## Next Steps

### Week 1: Learn and Validate

1. ✅ Complete quick start (done!)
2. Run backtests on different strategies
3. Review signal generation in Grafana
4. Understand feature engineering pipeline
5. Monitor paper trading performance

### Week 2-4: Paper Trading

1. Run paper trading continuously
2. Monitor daily P&L and Sharpe ratio
3. Test different symbols and strategies
4. Optimize position sizing
5. Track win rate and drawdown

### Month 2-3: Optimization

1. Retrain models weekly with new data
2. A/B test different strategies
3. Improve feature engineering
4. Optimize risk management parameters
5. Aim for consistent Sharpe > 1.2

### Month 4-6: Preparation for Live

**Before going live, achieve:**
- ✅ 3+ months consistent paper trading profits
- ✅ Sharpe ratio > 1.2
- ✅ Max drawdown < 20%
- ✅ Win rate > 52%
- ✅ Understand every component

### Month 7+: Go Live

1. Deploy to AWS (optional, see `ENTERPRISE_ARCHITECTURE.md`)
2. Start with $10,000 real capital
3. Scale gradually based on performance
4. Monitor and optimize continuously

---

## Troubleshooting

### Services won't start

```bash
# Check Docker is running
docker ps

# Check logs
docker-compose -f docker-compose.local.yml logs

# Restart services
docker-compose -f docker-compose.local.yml restart
```

### Database connection error

```bash
# Check if TimescaleDB is ready
docker exec quantcli-timescaledb-local pg_isready -U quantcli

# Reinitialize database
python scripts/init_database.py
```

### API rate limits

```bash
# Check rate limit status
python scripts/update_data.py --stats

# Wait for rate limits to reset (usually 24 hours)
# Or add backup API keys in .env
```

### Out of memory

```bash
# Check memory usage
docker stats

# Reduce resource limits in docker-compose.local.yml
# Or close other applications
```

---

## Cost Analysis

### Local Development: ~$5/month

```
Hardware:         $0 (your computer)
Electricity:      ~$5/month
API Keys:         $0 (free tiers)
──────────────────────────────
Total:            ~$5/month
```

### When to Move to Production?

**Only when:**
- ✅ 3+ months profitable paper trading
- ✅ Consistent Sharpe ratio > 1.2
- ✅ Max drawdown < 20%
- ✅ Comfortable with system and risk
- ✅ Ready to deploy real capital

**Production cost:** ~$1,800/month (AWS infrastructure)

**But:** 15% annual return on $250k = $37,500/year
Profit after costs: $37,500 - $21,600 = $15,900/year

---

## Support

**Documentation:**
- `LOCAL_DEVELOPMENT.md` - Detailed local development guide
- `IMPLEMENTATION_GUIDE.md` - Complete code implementation guide
- `SIGNAL_GENERATION_PIPELINE.md` - How predictions are made
- `ENTERPRISE_ARCHITECTURE.md` - Production deployment (when ready)

**Need Help?**
1. Check logs: `docker-compose -f docker-compose.local.yml logs -f`
2. Review documentation in `docs/` directory
3. Check GitHub issues: https://github.com/KanaparthySaiSreekar/QuantCLI/issues

---

## Important Reminders

⚠️ **Paper Trading First**
- NEVER start with live trading
- Paper trade for at least 3-6 months
- Validate consistent profitability

⚠️ **Risk Management**
- Start small ($10k) when going live
- Never risk more than 2% per position
- Respect the daily loss limit
- Kill switches are your friend

⚠️ **India LRS Compliance**
- $250,000 annual limit
- Proper tax treatment (capital gains)
- Schedule FA reporting required
- Form 67 for foreign tax credit

---

**🎉 You're all set! Start with paper trading and prove profitability before going live.**

**Questions?** See `LOCAL_DEVELOPMENT.md` for detailed information.
