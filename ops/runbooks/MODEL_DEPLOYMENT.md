# Model Deployment Runbook

**Purpose:** Step-by-step procedures for deploying ML models to production

**Owner:** ML Team
**Last Updated:** 2025-11-13

---

## Pre-Deployment Checklist

Before deploying any model to production, ensure ALL of these are complete:

- [ ] **CPCV Validation Passed** (Sharpe > 1.2, PSR > 0.95, Drawdown < 20%)
- [ ] **ONNX Parity Test Passed** (accuracy loss < 1%)
- [ ] **Model Registered in MLflow** with dataset hash and version
- [ ] **Feature Version Documented** in DataHub/MLflow
- [ ] **Canary Paper Trade Completed** (24h shadow routing, acceptable metrics)
- [ ] **Automated Tests Passed** (unit + integration + parity tests)
- [ ] **Security Scan Passed** (model artifacts signed, checksummed)
- [ ] **Governance Approval** (if required for large capital)

**DO NOT deploy if ANY of these fail.**

---

## Deployment Process

### Step 1: Validate Model in Staging

```bash
# Load model from staging
python -c "
from src.ml.training.mlflow_trainer import MLflowTrainer
trainer = MLflowTrainer()
model = trainer.load_production_model('ensemble_v1', stage='Staging')
print(f'Loaded model: {model}')
"

# Run ONNX parity test
python scripts/validate_onnx.py \
    --model models/production/ensemble_v1_fp32.onnx \
    --test-data data/test/X_test.npy \
    --threshold 0.01

# Expected: ✅ Parity test PASSED
```

### Step 2: Canary Deployment (Shadow Mode)

```bash
# Deploy to canary environment (paper trading, shadow routing)
# Model serves predictions but doesn't execute trades

# Start canary with 10% traffic
kubectl apply -f k8s/canary/trading-engine-canary.yaml

# Monitor for 24 hours
python scripts/monitor_canary.py \
    --duration 24h \
    --alert-threshold 0.05

# Metrics to watch:
# - Prediction latency (<200ms)
# - Fill rate mismatch (<2%)
# - P&L divergence from production (<5%)
# - Error rate (<0.1%)
```

### Step 3: Promote to Production

**Only proceed if canary metrics are acceptable.**

```python
# Promote model to production
from src.ml.training.mlflow_trainer import MLflowTrainer

trainer = MLflowTrainer()
success = trainer.promote_model_to_production(
    model_name='ensemble_v1',
    version=None,  # Latest in staging
)

if success:
    print("✅ Model promoted to Production")
else:
    print("❌ Promotion failed")
    exit(1)
```

### Step 4: Blue-Green Deployment

```bash
# Deploy new version (green) alongside current (blue)
kubectl apply -f k8s/production/trading-engine-green.yaml

# Wait for green to be ready
kubectl wait --for=condition=ready pod -l version=green -n quantcli

# Gradually shift traffic: 10% → 50% → 100%
kubectl patch service trading-engine -p '{"spec":{"selector":{"version":"green","traffic":"10%"}}}'

# Monitor for 1 hour
sleep 3600

# If metrics are good, shift to 50%
kubectl patch service trading-engine -p '{"spec":{"selector":{"version":"green","traffic":"50%"}}}'

# Monitor for 1 hour
sleep 3600

# Full cutover
kubectl patch service trading-engine -p '{"spec":{"selector":{"version":"green"}}}'

# Decommission blue after 24h of green stability
```

### Step 5: Post-Deployment Validation

```bash
# Verify production model is correct version
python scripts/verify_production_model.py

# Check metrics dashboard
open http://grafana:3000/d/trading-performance

# Verify immutable tagging on orders
psql -d quantcli -c "
SELECT DISTINCT metadata->>'model_version', COUNT(*)
FROM orders
WHERE time > NOW() - INTERVAL '1 hour'
GROUP BY metadata->>'model_version';
"

# Expected: Only new model version
```

---

## Rollback Procedure

**Trigger:** If production metrics degrade OR kill switch triggers

### Immediate Rollback (< 5 minutes)

```bash
# Emergency rollback to previous version
kubectl patch service trading-engine -p '{"spec":{"selector":{"version":"blue"}}}'

# OR

# Rollback in MLflow
python scripts/rollback_model.py \
    --model-name ensemble_v1 \
    --stage Production \
    --previous-version

# Verify rollback
python scripts/verify_production_model.py

# Alert team
python scripts/send_alert.py \
    --severity CRITICAL \
    --message "Model rolled back due to performance degradation"
```

### Post-Rollback Analysis

1. **Export production logs**
   ```bash
   kubectl logs -l app=trading-engine --since=1h > /tmp/rollback_logs.txt
   ```

2. **Analyze failed predictions**
   ```sql
   SELECT * FROM predictions
   WHERE time > NOW() - INTERVAL '1 hour'
     AND model_name = 'ensemble_v1_new'
   ORDER BY confidence ASC
   LIMIT 100;
   ```

3. **Check for drift**
   ```python
   python scripts/analyze_drift.py \
       --baseline data/baseline_features.parquet \
       --current data/production_features_today.parquet
   ```

4. **Root cause analysis**
   - Data quality issues?
   - Feature engineering bug?
   - Model overfitting?
   - Infrastructure issue?

5. **Document findings** in incident report

---

## Model Tagging Requirements

**Every order MUST include these tags:**

```python
order_metadata = {
    "model_version": "ensemble_v1.23",
    "model_run_id": "a1b2c3d4e5f6",
    "feature_version": "v1.2.0",
    "dataset_manifest_id": "sha256:abc123...",
    "onnx_model": "ensemble_v1_int8.onnx",
    "inference_latency_ms": 145,
    "deployed_at": "2025-11-13T10:30:00Z",
}

# Tag order
cursor.execute("""
    INSERT INTO orders (order_id, ..., metadata)
    VALUES (%s, ..., %s)
""", (order_id, ..., json.dumps(order_metadata)))
```

**Purpose:** Audit trail for compliance and debugging

---

## Monitoring After Deployment

**Monitor these metrics for 7 days:**

1. **Performance Metrics** (Grafana dashboard)
   - Sharpe ratio (rolling 30d)
   - Max drawdown
   - Win rate
   - P&L vs benchmark

2. **System Metrics**
   - Model inference latency (<200ms)
   - Feature store hit rate (>99%)
   - Order ACK latency (<100ms)
   - Error rate (<0.1%)

3. **Drift Metrics** (daily job)
   - PSI per feature
   - KS test p-values
   - Feature distribution plots

4. **Compliance Metrics**
   - All orders tagged correctly
   - Audit trail complete
   - Pre-trade checks passing (100%)

**Alert Thresholds:**

- **CRITICAL:** Sharpe < 0.8 OR Drawdown > -25% OR Error rate > 1%
- **WARNING:** Sharpe < 1.0 OR Drawdown > -20% OR Drift detected in >5 features
- **INFO:** Performance within expected range

---

## Automated Rollback Policy

**Auto-rollback triggers:**

1. **Performance:** Sharpe falls below 0.8 for 3 consecutive days
2. **Drawdown:** Exceeds -25% at any time
3. **Error Rate:** >1% of predictions fail
4. **Latency:** >500ms for >5% of requests
5. **Fill Rate:** Mismatch >5% vs paper trading

**Implementation:**

```python
# Automated monitor (runs every 5 minutes)
if production_sharpe < 0.8 and consecutive_days >= 3:
    trigger_auto_rollback()
    send_alert("AUTO-ROLLBACK: Sharpe degradation")

if current_drawdown < -0.25:
    trigger_auto_rollback()
    trigger_kill_switch()  # Stop all trading
    send_alert("AUTO-ROLLBACK + KILL SWITCH: Excessive drawdown")
```

---

## Governance & Approval

**Staging → Production promotion requires:**

- **Models <$50k capital:** ML Lead approval
- **Models $50k-$250k:** ML Lead + Risk Manager approval
- **Models >$250k:** Full governance board approval

**Approval checklist:**

- [ ] CPCV validation report reviewed
- [ ] Backtest results meet targets
- [ ] Risk limits configured correctly
- [ ] Compliance tags implemented
- [ ] Monitoring dashboards configured
- [ ] Rollback procedure tested

---

## Common Issues & Solutions

### Issue: ONNX Parity Test Fails

**Symptom:** Accuracy loss >1% between native and ONNX

**Solution:**
1. Check ONNX opset version (use 17)
2. Verify feature scaling is identical
3. Use FP32 instead of INT8 if accuracy critical
4. Re-export model with correct configuration

### Issue: Canary Metrics Diverge

**Symptom:** Canary P&L differs >5% from production

**Solution:**
1. Check feature store - are features identical?
2. Verify same pre-trade checks applied
3. Check for timing issues (staleness)
4. Compare order tags - same execution logic?

### Issue: Model Latency Spikes

**Symptom:** Inference >200ms

**Solution:**
1. Check ONNX model is loaded (not native)
2. Verify INT8 quantization active
3. Check for resource contention (CPU/memory)
4. Scale horizontally (more pods)

---

## Contact

**Escalation:**
- ML Team Lead: ml-lead@quantcli.com
- Risk Manager: risk@quantcli.com
- On-call: oncall@quantcli.com (24/7)

**Resources:**
- Grafana: http://grafana:3000/d/trading-performance
- MLflow: http://mlflow:5000
- Runbooks: /ops/runbooks/
- Incident Reports: /ops/incidents/

---

**Remember:** When in doubt, ROLLBACK. Better to be cautious than lose capital.
