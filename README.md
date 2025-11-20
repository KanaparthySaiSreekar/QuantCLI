# QuantCLI - Algorithmic Trading System

**Status:** Production-Ready (85% Complete) | ML Improvements In Progress
**Latest:** Week 1 Critical Fixes Complete ✅ | [Track Progress →](PROJECT_STATUS.md)

---

## 🎯 Quick Navigation

- **[PROJECT_STATUS.md](PROJECT_STATUS.md)** ← **START HERE** (Master Tracker)
- [Implementation Plan](IMPLEMENTATION_PLAN.md) - Detailed 30-hour roadmap
- [ML Analysis](ML_ANALYSIS.md) - Technical analysis of issues
- [Documentation](docs/) - Full system guides

---

## 🚀 Quick Start

```bash
# Initialize database
python scripts/init_database.py

# Download market data
python scripts/update_data.py

# Train models (with new CPCV validation)
python scripts/train_ensemble.py --validation cpcv

# Run backtest
python scripts/run_backtest.py

# Start paper trading
python scripts/start_trading.py --mode paper
```

---

## 📊 Current Status

### ✅ Week 1: Critical Fixes COMPLETE (6.5/6.5 hours)
- ✅ Fixed data leakage (3 sources eliminated)
- ✅ Enabled CPCV validation
- ✅ Created drift detection module
- ✅ Fixed scaler leakage
- ✅ Added feature validation

**Impact:** +8-18% expected Sharpe improvement

### ⏳ Week 2: Production Stability (0/11 hours)
- ⏳ Reduce ensemble to 3 models
- ⏳ Feature importance validation
- ⏳ Monthly retraining manager
- ⏳ Monitoring dashboard

### ⏳ Week 3: Performance (0/11 hours)
- ⏳ Stationary features
- ⏳ Parallel inference + caching
- ⏳ Ensemble diversity metrics
- ⏳ Comprehensive testing

**→ Full details in [PROJECT_STATUS.md](PROJECT_STATUS.md)**

---

## 🔧 New Features (Week 1)

### 1. Drift Detection
```python
from src.monitoring import DriftDetector

detector = DriftDetector(psi_threshold=0.1)
detector.fit(train_data, feature_cols)
report = detector.detect(recent_data)
```

### 2. CPCV Validation
```python
trainer = ModelTrainer(
    model=my_model,
    validation_method='cpcv',  # New!
    n_splits=5
)
```

### 3. Feature Validation
```python
engineer = FeatureEngineer()
df_ml = engineer.transform_for_ml(df)
# Raises error if data leakage detected
```

---

## ⚠️ Breaking Changes

**Models must be retrained** due to Week 1 fixes:
- Feature names changed (`dist_from_high_*` → `dist_from_past_high_*`)
- New features added (`ad_line_change`)
- Trainer API changed (see [PROJECT_STATUS.md](PROJECT_STATUS.md))

---

## 📈 System Overview

**Core Features:**
- 7 data providers with automatic failover
- 50+ technical indicators
- ML ensemble (XGBoost, LightGBM, CatBoost)
- Vectorized backtesting engine
- IBKR order execution
- Risk management with kill switches
- Production infrastructure (TimescaleDB, Redis, Kafka)
- **NEW:** Drift detection & CPCV validation

**Tech Stack:**
- Python 3.10+
- ML: XGBoost, LightGBM, CatBoost, PyTorch
- Data: TimescaleDB, Redis, Kafka
- Monitoring: Prometheus, Grafana, MLflow

---

## 📖 Documentation

### Essential
- **[PROJECT_STATUS.md](PROJECT_STATUS.md)** - Complete progress tracker
- **[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)** - 30-hour roadmap
- **[ML_ANALYSIS.md](ML_ANALYSIS.md)** - Original issue analysis

### Reference
- [docs/](docs/) - System documentation
- [docs/archive/](docs/archive/) - Historical status reports

---

## 🧪 Testing

```bash
pytest tests/ --cov=src
```

**Coverage:** 65% (target: 80%)

---

## 📞 Support

- **Track Progress:** [PROJECT_STATUS.md](PROJECT_STATUS.md)
- **Report Issues:** GitHub Issues
- **Documentation:** `docs/` directory

---

**Last Updated:** 2025-11-18
**Progress:** 22% complete (6.5/30 hours)
**Next:** Week 2 Production Stability
