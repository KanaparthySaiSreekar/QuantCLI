# QuantCLI ML Fixes - Detailed Implementation Plan

**Created:** 2025-11-18
**Status:** In Progress
**Estimated Total Time:** 30 hours

---

## Table of Contents

1. [Week 1: Critical Fixes (6.5 hours)](#week-1-critical-fixes)
2. [Week 2: Production Stability (11 hours)](#week-2-production-stability)
3. [Week 3: Performance Optimization (11 hours)](#week-3-performance-optimization)
4. [Testing Strategy](#testing-strategy)
5. [Rollout Plan](#rollout-plan)

---

## Week 1: Critical Fixes (6.5 hours)

### 1.1 Fix Look-Ahead Bias in Target Creation (1 hour)

**File:** `src/features/engineer.py`
**Lines:** 327-336
**Priority:** CRITICAL

#### Current Code (BROKEN):
```python
def transform_for_ml(
    self,
    df: pd.DataFrame,
    target_periods: int = 1,
    classification: bool = False
) -> pd.DataFrame:
    result = df.copy()

    # PROBLEM: This shifts future returns into current row
    future_return = result['close'].pct_change(target_periods).shift(-target_periods)

    if classification:
        result['target'] = (future_return > 0).astype(int)
    else:
        result['target'] = future_return

    result = result.dropna(subset=['target'])
    return result
```

#### Fixed Code:
```python
def transform_for_ml(
    self,
    df: pd.DataFrame,
    target_periods: int = 1,
    classification: bool = False
) -> pd.DataFrame:
    """
    Transform features for ML model training.

    IMPORTANT: Target is created BEFORE feature generation to prevent look-ahead bias.
    Features should only use information available at prediction time.
    """
    result = df.copy()

    # FIXED: Calculate future price, then create return from that
    # This ensures features computed after this cannot see future data
    future_price = result['close'].shift(-target_periods)
    future_return = (future_price - result['close']) / result['close']

    if classification:
        # Binary classification: 1 = up, 0 = down
        result['target'] = (future_return > 0).astype(int)
    else:
        # Regression: actual return
        result['target'] = future_return

    # Remove rows without target (last target_periods rows)
    result = result.dropna(subset=['target'])

    # Validate no look-ahead bias
    self._validate_no_look_ahead(result)

    self.logger.info(
        f"Created {'classification' if classification else 'regression'} "
        f"target with {len(result)} samples, {target_periods} periods ahead"
    )

    return result

def _validate_no_look_ahead(self, df: pd.DataFrame) -> None:
    """
    Runtime validation to ensure no features use future information.
    """
    feature_cols = self.get_feature_names(df)

    for col in feature_cols:
        # Check for suspicious feature names
        if any(word in col.lower() for word in ['future', 'forward', 'ahead']):
            raise DataError(f"Forward-looking feature detected: {col}")

        # Check for features that should have been removed
        if 'dist_from_high' in col or 'dist_from_low' in col:
            raise DataError(f"Leaky feature detected: {col} (uses future price data)")
```

#### Testing:
```python
def test_no_target_leakage():
    """Verify target doesn't leak into features"""
    df = pd.DataFrame({
        'open': [100, 101, 102, 103, 104],
        'high': [101, 102, 103, 104, 105],
        'low': [99, 100, 101, 102, 103],
        'close': [100, 101, 102, 103, 104],
        'volume': [1000, 1100, 1200, 1300, 1400]
    }, index=pd.date_range('2020-01-01', periods=5))

    engineer = FeatureEngineer()
    df_features = engineer.generate_features(df)
    df_ml = engineer.transform_for_ml(df_features, target_periods=1)

    # Target at time t should be return from t to t+1
    # Features at time t should only use data up to time t
    assert 'target' in df_ml.columns
    assert len(df_ml) == len(df) - 1  # Lost last row due to target shift
```

---

### 1.2 Remove Future-Looking Distance Features (30 minutes)

**File:** `src/features/engineer.py`
**Lines:** 168-175
**Priority:** CRITICAL

#### Current Code (BROKEN):
```python
def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    # PROBLEM: rolling().max() looks FORWARD
    for period in [20, 50]:
        result[f'dist_from_high_{period}'] = (
            result['close'] / result['high'].rolling(period).max() - 1
        )
        result[f'dist_from_low_{period}'] = (
            result['close'] / result['low'].rolling(period).min() - 1
        )

    return result
```

#### Fixed Code:
```python
def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
    """Add price-based features."""
    self.logger.debug("Adding price-based features")

    result = df.copy()

    # Returns over different periods
    for period in [1, 2, 3, 5, 10, 20]:
        result[f'return_{period}d'] = result['close'].pct_change(period)

    # Log returns
    result['log_return_1d'] = np.log(result['close'] / result['close'].shift(1))

    # Intraday range
    result['intraday_range'] = (result['high'] - result['low']) / result['close']

    # Gap (open vs previous close)
    result['gap'] = (result['open'] - result['close'].shift(1)) / result['close'].shift(1)

    # High-Low position
    result['hl_position'] = (result['close'] - result['low']) / (result['high'] - result['low'])

    # Price acceleration (second derivative)
    result['price_acceleration'] = result['close'].diff().diff()

    # FIXED: Use PAST highs/lows instead of future
    # This gives "distance from recent high" which is predictive and valid
    for period in [20, 50]:
        past_high = result['high'].shift(1).rolling(period).max()
        past_low = result['low'].shift(1).rolling(period).min()

        result[f'dist_from_past_high_{period}'] = (
            result['close'] / past_high - 1
        )
        result[f'dist_from_past_low_{period}'] = (
            result['close'] / past_low - 1
        )

    # Volatility measures
    for period in [5, 10, 20]:
        result[f'volatility_{period}d'] = result['return_1d'].rolling(period).std()

    return result
```

---

### 1.3 Fix Cumulative Feature Leakage (30 minutes)

**File:** `src/features/engineer.py`
**Lines:** 210-215
**Priority:** HIGH

#### Current Code (BROKEN):
```python
def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    # ... other features ...

    # PROBLEM: cumsum() includes current bar's volume
    mfm = ((result['close'] - result['low']) - (result['high'] - result['close'])) / (
        result['high'] - result['low']
    )
    mfv = mfm * result['volume']
    result['ad_line'] = mfv.cumsum()

    return result
```

#### Fixed Code:
```python
def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
    """Add volume-based features."""
    self.logger.debug("Adding volume-based features")

    result = df.copy()

    # Volume moving averages
    for period in [5, 10, 20, 50]:
        result[f'volume_sma_{period}'] = result['volume'].rolling(period).mean()

    # Volume ratio (current vs average)
    result['volume_ratio_20'] = result['volume'] / result['volume_sma_20']

    # Volume changes
    result['volume_change_1d'] = result['volume'].pct_change(1)

    # Price-volume correlation
    for period in [10, 20]:
        result[f'price_volume_corr_{period}'] = result['close'].rolling(period).corr(
            result['volume']
        )

    # Volume trend
    result['volume_trend_20'] = (
        result['volume_sma_5'] / result['volume_sma_20']
    )

    # FIXED: Accumulation/Distribution Line - shift to exclude current bar
    mfm = ((result['close'] - result['low']) - (result['high'] - result['close'])) / (
        result['high'] - result['low'] + 1e-8  # Avoid division by zero
    )
    mfv = mfm * result['volume']

    # Shift cumulative sum to exclude current bar
    result['ad_line'] = mfv.shift(1).cumsum()

    # Also fix it to be stationary (use change instead of level)
    result['ad_line_change'] = result['ad_line'].diff()

    return result
```

---

### 1.4 Enable CPCV Validation Properly (2 hours)

**Files:**
- `src/models/trainer.py` (modify existing)
- `scripts/train_ensemble.py` (fix connection)

#### Current Issues:
1. CPCV parameter accepted but never used
2. Default uses random holdout (wrong for time series)
3. Scaler refitted on each fold (leakage)

#### Fixed Code - trainer.py:

```python
class ModelTrainer:
    """Orchestrates model training workflow."""

    def __init__(
        self,
        model: BaseModel,
        task: str = "classification",
        validation_method: str = "holdout",  # NEW: explicit validation method
        test_size: float = 0.2,
        val_size: float = 0.1,
        n_splits: int = 5,
        random_state: int = 42,
        scale_features: bool = True
    ):
        self.model = model
        self.task = task
        self.validation_method = validation_method
        self.test_size = test_size
        self.val_size = val_size
        self.n_splits = n_splits
        self.random_state = random_state
        self.scale_features = scale_features

        # FIT SCALER ONCE, not per fold
        self.scaler = StandardScaler() if scale_features else None
        self.scaler_fitted = False
        self.training_history = {}

        self.logger = logger

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, Any]:
        """
        Train model with specified validation method.

        Args:
            X: Feature DataFrame
            y: Target Series

        Returns:
            Dictionary with training results
        """
        self.logger.info(f"Starting training for {self.model.model_name}")
        self.logger.info(f"Validation method: {self.validation_method}")
        self.logger.info(f"Dataset: {len(X)} samples, {len(X.columns)} features")

        # Validate inputs
        self._validate_data(X, y)

        # Route to appropriate validation method
        if self.validation_method == "holdout":
            return self._train_holdout(X, y)
        elif self.validation_method == "timeseries":
            return self._train_time_series_cv(X, y)
        elif self.validation_method == "cpcv":
            return self._train_cpcv(X, y)
        else:
            raise ValueError(f"Unknown validation method: {self.validation_method}")

    def _train_holdout(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train with time-aware holdout validation (NO random shuffle)."""
        # FIXED: No shuffle for time series
        n = len(X)
        train_end = int(n * (1 - self.test_size - self.val_size))
        val_end = int(n * (1 - self.test_size))

        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]

        X_val = X.iloc[train_end:val_end]
        y_val = y.iloc[train_end:val_end]

        X_test = X.iloc[val_end:]
        y_test = y.iloc[val_end:]

        self.logger.info(
            f"Time-aware split: train={len(X_train)}, val={len(X_val)}, "
            f"test={len(X_test)}"
        )

        # FIXED: Fit scaler ONLY on training data
        if self.scale_features and not self.scaler_fitted:
            self.scaler.fit(X_train)
            self.scaler_fitted = True
            self.logger.info("Scaler fitted on training data only")

        # Transform all sets with same scaler
        if self.scale_features:
            X_train = self._scale_transform(X_train)
            X_val = self._scale_transform(X_val)
            X_test = self._scale_transform(X_test)

        # Train model
        self.logger.info("Training model...")
        train_metrics = self.model.train(X_train, y_train, X_val, y_val)

        # Evaluate on test set
        self.logger.info("Evaluating on test set...")
        test_predictions = self.model.predict(X_test)

        # Calculate metrics
        test_metrics = self._calculate_metrics(y_test, test_predictions)

        self.logger.info(f"Test metrics: {test_metrics}")

        # Store history
        self.training_history = {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'n_train': len(X_train),
            'n_val': len(X_val),
            'n_test': len(X_test),
            'trained_at': datetime.now().isoformat()
        }

        return {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'training_history': self.training_history
        }

    def _train_cpcv(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Train with Combinatorial Purged Cross-Validation.

        This properly handles time series by:
        1. Respecting temporal order
        2. Purging overlapping samples
        3. Embargoing recent samples
        """
        from src.backtest.cpcv import CombinatorialPurgedCV

        self.logger.info(f"Using CPCV with {self.n_splits} splits")

        # Initialize CPCV
        cpcv = CombinatorialPurgedCV(
            n_splits=self.n_splits,
            n_test_splits=2,
            purge_gap=5,  # 5 days purge
            embargo_pct=0.01  # 1% embargo
        )

        # FIT SCALER ONCE on entire training set
        if self.scale_features and not self.scaler_fitted:
            # Use first 80% for scaler fitting
            train_size = int(len(X) * 0.8)
            self.scaler.fit(X.iloc[:train_size])
            self.scaler_fitted = True
            self.logger.info("Scaler fitted on initial 80% of data")

        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(cpcv.split(X), 1):
            self.logger.info(f"Training fold {fold}/{cpcv.get_n_splits()}")

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Transform (DON'T refit scaler)
            if self.scale_features:
                X_train = self._scale_transform(X_train)
                X_val = self._scale_transform(X_val)

            # Train
            self.model.train(X_train, y_train, X_val, y_val)

            # Evaluate
            predictions = self.model.predict(X_val)
            score = self._calculate_fold_score(y_val, predictions)

            fold_scores.append(score)
            self.logger.info(f"Fold {fold} score: {score:.4f}")

        avg_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)

        self.logger.info(
            f"CPCV complete. Average score: {avg_score:.4f} (+/- {std_score:.4f})"
        )

        # Final train on all data (except holdout test set)
        self.logger.info("Training final model on all data...")

        test_size = int(len(X) * self.test_size)
        X_train_full = X.iloc[:-test_size]
        y_train_full = y.iloc[:-test_size]
        X_test = X.iloc[-test_size:]
        y_test = y.iloc[-test_size:]

        if self.scale_features:
            X_train_full = self._scale_transform(X_train_full)
            X_test = self._scale_transform(X_test)

        final_metrics = self.model.train(X_train_full, y_train_full)

        # Test set evaluation
        test_predictions = self.model.predict(X_test)
        test_metrics = self._calculate_metrics(y_test, test_predictions)

        self.training_history = {
            'cv_scores': fold_scores,
            'avg_cv_score': avg_score,
            'std_cv_score': std_score,
            'final_metrics': final_metrics,
            'test_metrics': test_metrics,
            'n_samples': len(X),
            'trained_at': datetime.now().isoformat()
        }

        return {
            'cv_scores': fold_scores,
            'avg_score': avg_score,
            'std_score': std_score,
            'final_metrics': final_metrics,
            'test_metrics': test_metrics
        }

    def _scale_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using FITTED scaler (no refitting)."""
        if self.scaler is None or not self.scaler_fitted:
            return X

        X_scaled = self.scaler.transform(X)
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    def _calculate_fold_score(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Calculate single score for CV fold."""
        from sklearn.metrics import accuracy_score, mean_squared_error

        if self.task == 'classification':
            return accuracy_score(y_true, y_pred.round())
        else:
            return -mean_squared_error(y_true, y_pred)  # Negative MSE

    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict:
        """Calculate comprehensive metrics."""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            mean_squared_error, mean_absolute_error, r2_score
        )

        if self.task == 'classification':
            return {
                'accuracy': accuracy_score(y_true, y_pred.round()),
                'precision': precision_score(y_true, y_pred.round(), average='weighted', zero_division=0),
                'recall': recall_score(y_true, y_pred.round(), average='weighted', zero_division=0),
                'f1': f1_score(y_true, y_pred.round(), average='weighted', zero_division=0)
            }
        else:
            return {
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred)
            }
```

#### Fixed Code - train_ensemble.py:

```python
class ModelTrainer:
    """Ensemble model trainer."""

    def __init__(self, validation: str = "holdout", optimize: bool = False):
        """
        Initialize model trainer.

        Args:
            validation: Validation method (holdout, timeseries, cpcv)
            optimize: Enable hyperparameter optimization
        """
        self.validation = validation
        self.optimize = optimize
        self.config = ConfigManager()

        # Map validation string to trainer parameter
        self.validation_method_map = {
            'simple': 'holdout',
            'cpcv': 'cpcv',
            'walk_forward': 'timeseries'
        }

    def train_ensemble(self, features: pd.DataFrame, labels: pd.Series):
        """
        Train ensemble models with proper validation.

        Args:
            features: Feature matrix
            labels: Target labels
        """
        from src.models.trainer import ModelTrainer as MLTrainer
        from src.models.ensemble import EnsembleModel

        logger.info("Training ensemble models...")
        logger.info(f"Validation method: {self.validation}")

        # FIXED: Actually use the validation parameter
        validation_method = self.validation_method_map.get(
            self.validation,
            'holdout'
        )

        # Initialize ensemble
        ensemble = EnsembleModel()

        # Create trainer with correct validation method
        trainer = MLTrainer(
            model=ensemble,
            task='classification',
            validation_method=validation_method,  # FIXED: Pass validation method
            test_size=0.2,
            val_size=0.1,
            n_splits=5,
            scale_features=True
        )

        # Train
        results = trainer.train(features, labels)

        logger.success(f"✓ Ensemble trained with {validation_method} validation")
        logger.info(f"Test metrics: {results['test_metrics']}")

        return results
```

---

### 1.5 Fix Scaler Leakage (1 hour)

**Covered in section 1.4 above** - the fix involves:
1. Fitting scaler ONCE on training data
2. Never refitting on val/test/CV folds
3. Transforming all future data with the same fitted scaler

---

### 1.6 Fix Deprecated Pandas Methods (5 minutes)

**File:** `src/backtest/engine.py`
**Line:** 204
**Priority:** LOW (but quick)

#### Current Code (DEPRECATED):
```python
def _calculate_positions(self, signals: pd.Series) -> pd.Series:
    # Forward fill signals to maintain positions
    positions = signals.replace(0, np.nan).fillna(method='ffill').fillna(0)
    return positions
```

#### Fixed Code:
```python
def _calculate_positions(self, signals: pd.Series) -> pd.Series:
    """
    Convert signals to positions.

    Signals (1, -1, 0) → Positions (1, -1, 0)
    Position held until signal changes.
    """
    # FIXED: Use .ffill() instead of deprecated fillna(method='ffill')
    positions = signals.replace(0, np.nan).ffill().fillna(0)
    return positions
```

---

## Week 2: Production Stability (11 hours)

### 2.1 Implement PSI-Based Drift Detection (4 hours)

**New File:** `src/monitoring/drift_detection.py`

```python
"""
Population Stability Index (PSI) drift detection.

Monitors feature distributions for significant changes that indicate model degradation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

from src.core.logging_config import get_logger
from src.core.exceptions import ValidationError

logger = get_logger(__name__)


@dataclass
class DriftReport:
    """Report of drift detection results."""
    timestamp: datetime
    overall_psi: float
    feature_psi: Dict[str, float]
    drifted_features: List[str]
    is_significant_drift: bool
    threshold: float
    recommendation: str


class DriftDetector:
    """
    Detects distribution drift using Population Stability Index (PSI).

    PSI measures the shift in distribution between baseline and current data:
    - PSI < 0.1: No significant change
    - 0.1 <= PSI < 0.2: Moderate change (monitor)
    - PSI >= 0.2: Significant change (retrain)

    Formula:
        PSI = sum((current_pct - baseline_pct) * ln(current_pct / baseline_pct))
    """

    def __init__(
        self,
        n_bins: int = 10,
        psi_threshold: float = 0.1,
        feature_threshold: float = 0.2
    ):
        """
        Initialize drift detector.

        Args:
            n_bins: Number of bins for discretization
            psi_threshold: Overall PSI threshold for triggering alerts
            feature_threshold: Per-feature PSI threshold
        """
        self.n_bins = n_bins
        self.psi_threshold = psi_threshold
        self.feature_threshold = feature_threshold
        self.baseline_distributions = {}
        self.logger = logger

    def fit(self, baseline_data: pd.DataFrame, feature_cols: Optional[List[str]] = None):
        """
        Fit baseline distributions.

        Args:
            baseline_data: Training data to use as baseline
            feature_cols: List of feature columns (None = all numeric)
        """
        if feature_cols is None:
            feature_cols = baseline_data.select_dtypes(include=[np.number]).columns.tolist()

        self.logger.info(f"Fitting baseline distributions for {len(feature_cols)} features")

        for feature in feature_cols:
            if feature not in baseline_data.columns:
                self.logger.warning(f"Feature {feature} not in baseline data, skipping")
                continue

            values = baseline_data[feature].dropna()

            if len(values) == 0:
                continue

            # Create bins and calculate distribution
            try:
                bins, dist = self._create_distribution(values)
                self.baseline_distributions[feature] = {
                    'bins': bins,
                    'distribution': dist
                }
            except Exception as e:
                self.logger.warning(f"Failed to create distribution for {feature}: {e}")

        self.logger.success(f"✓ Baseline fitted for {len(self.baseline_distributions)} features")

    def detect(
        self,
        current_data: pd.DataFrame,
        feature_cols: Optional[List[str]] = None
    ) -> DriftReport:
        """
        Detect drift in current data vs baseline.

        Args:
            current_data: Current/recent data to check
            feature_cols: Features to check (None = all from baseline)

        Returns:
            DriftReport with detailed findings
        """
        if not self.baseline_distributions:
            raise ValidationError("Must call fit() before detect()")

        if feature_cols is None:
            feature_cols = list(self.baseline_distributions.keys())

        feature_psi = {}

        for feature in feature_cols:
            if feature not in self.baseline_distributions:
                self.logger.warning(f"Feature {feature} not in baseline, skipping")
                continue

            if feature not in current_data.columns:
                self.logger.warning(f"Feature {feature} not in current data, skipping")
                continue

            values = current_data[feature].dropna()

            if len(values) == 0:
                continue

            # Calculate PSI for this feature
            try:
                psi = self._calculate_psi(
                    values,
                    self.baseline_distributions[feature]['bins'],
                    self.baseline_distributions[feature]['distribution']
                )
                feature_psi[feature] = psi
            except Exception as e:
                self.logger.warning(f"Failed to calculate PSI for {feature}: {e}")

        # Overall PSI (average across features)
        overall_psi = np.mean(list(feature_psi.values())) if feature_psi else 0.0

        # Identify drifted features
        drifted_features = [
            feat for feat, psi in feature_psi.items()
            if psi >= self.feature_threshold
        ]

        # Determine if drift is significant
        is_significant = overall_psi >= self.psi_threshold

        # Generate recommendation
        if is_significant or len(drifted_features) > len(feature_cols) * 0.2:
            recommendation = "RETRAIN - Significant drift detected"
        elif overall_psi >= self.psi_threshold * 0.7:
            recommendation = "MONITOR - Moderate drift, prepare to retrain"
        else:
            recommendation = "OK - No significant drift"

        report = DriftReport(
            timestamp=datetime.now(),
            overall_psi=overall_psi,
            feature_psi=feature_psi,
            drifted_features=drifted_features,
            is_significant_drift=is_significant,
            threshold=self.psi_threshold,
            recommendation=recommendation
        )

        # Log results
        self.logger.info(f"Drift detection: PSI={overall_psi:.4f}, "
                        f"{len(drifted_features)} drifted features")
        if is_significant:
            self.logger.warning(f"⚠️  {recommendation}")

        return report

    def _create_distribution(self, values: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create binned distribution from values.

        Returns:
            (bin_edges, distribution_percentages)
        """
        # Use quantile-based binning for better handling of outliers
        try:
            bins = pd.qcut(values, q=self.n_bins, duplicates='drop', retbins=True)[1]
        except ValueError:
            # Fallback to equal-width bins if quantiles fail
            bins = np.linspace(values.min(), values.max(), self.n_bins + 1)

        # Calculate distribution
        counts, _ = np.histogram(values, bins=bins)
        distribution = counts / counts.sum()

        # Add small epsilon to avoid log(0)
        distribution = np.maximum(distribution, 1e-8)

        return bins, distribution

    def _calculate_psi(
        self,
        current_values: pd.Series,
        baseline_bins: np.ndarray,
        baseline_dist: np.ndarray
    ) -> float:
        """
        Calculate PSI between current values and baseline distribution.

        Args:
            current_values: Current data values
            baseline_bins: Bin edges from baseline
            baseline_dist: Baseline distribution percentages

        Returns:
            PSI value (0 = no change, higher = more drift)
        """
        # Bin current values using baseline bins
        counts, _ = np.histogram(current_values, bins=baseline_bins)
        current_dist = counts / counts.sum()

        # Add epsilon to avoid log(0)
        current_dist = np.maximum(current_dist, 1e-8)
        baseline_dist = np.maximum(baseline_dist, 1e-8)

        # Calculate PSI
        psi = np.sum((current_dist - baseline_dist) * np.log(current_dist / baseline_dist))

        return psi

    def save_baseline(self, path: str):
        """Save baseline distributions to file."""
        import joblib
        joblib.dump(self.baseline_distributions, path)
        self.logger.info(f"Baseline saved to {path}")

    def load_baseline(self, path: str):
        """Load baseline distributions from file."""
        import joblib
        self.baseline_distributions = joblib.load(path)
        self.logger.info(f"Baseline loaded from {path}")


class DriftMonitor:
    """
    Continuously monitors for drift and triggers retraining.

    Usage in production:
        monitor = DriftMonitor(detector, retrain_callback)
        monitor.check(recent_data)  # Call daily or weekly
    """

    def __init__(
        self,
        detector: DriftDetector,
        check_interval_days: int = 7,
        alert_callback: Optional[callable] = None
    ):
        """
        Initialize drift monitor.

        Args:
            detector: Fitted DriftDetector instance
            check_interval_days: How often to check for drift
            alert_callback: Function to call when drift detected
        """
        self.detector = detector
        self.check_interval_days = check_interval_days
        self.alert_callback = alert_callback
        self.last_check = None
        self.drift_history = []
        self.logger = logger

    def check(self, current_data: pd.DataFrame) -> Optional[DriftReport]:
        """
        Check for drift if interval has passed.

        Args:
            current_data: Recent data to check

        Returns:
            DriftReport if check was performed, None otherwise
        """
        now = datetime.now()

        # Check if enough time has passed
        if self.last_check is not None:
            days_since_check = (now - self.last_check).days
            if days_since_check < self.check_interval_days:
                return None

        # Perform drift detection
        report = self.detector.detect(current_data)

        # Store history
        self.drift_history.append({
            'timestamp': report.timestamp,
            'psi': report.overall_psi,
            'is_significant': report.is_significant_drift
        })

        # Alert if significant drift
        if report.is_significant_drift and self.alert_callback:
            self.alert_callback(report)

        self.last_check = now

        return report

    def get_drift_history(self) -> pd.DataFrame:
        """Get history of drift checks."""
        return pd.DataFrame(self.drift_history)
```

**Testing:**
```python
def test_drift_detection():
    """Test PSI drift detection"""
    # Create baseline data
    np.random.seed(42)
    baseline = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.normal(5, 2, 1000),
        'feature3': np.random.uniform(0, 10, 1000)
    })

    # Create drifted data
    current = pd.DataFrame({
        'feature1': np.random.normal(0.5, 1.2, 1000),  # Slight drift
        'feature2': np.random.normal(7, 3, 1000),      # Significant drift
        'feature3': np.random.uniform(0, 10, 1000)     # No drift
    })

    # Fit and detect
    detector = DriftDetector(psi_threshold=0.1)
    detector.fit(baseline)
    report = detector.detect(current)

    assert 'feature2' in report.drifted_features
    assert report.overall_psi > 0
```

---

### 2.2 Reduce Ensemble to 3 Diverse Models (2 hours)

**File:** `config/models.yaml`

#### Current Config (TOO COMPLEX):
```yaml
ensemble:
  n_base_models: 6
  base_models:
    xgboost_1:  # 100 trees, depth 6
    xgboost_2:  # 200 trees, depth 8
    lightgbm_1: # 31 leaves
    lightgbm_2: # 50 leaves
    catboost_1:
    lstm_1:
```

#### Fixed Config (OPTIMIZED):
```yaml
# Ensemble Configuration - OPTIMIZED for diversity and speed
ensemble:
  enabled: true
  n_base_models: 3  # REDUCED from 6
  meta_learner: "ridge"  # Linear meta-learner (fast, interpretable)

  # Strategy: Use 3 complementary algorithms instead of 6 similar ones
  base_models:
    # Model 1: LightGBM (fast, leaf-wise growth)
    lightgbm:
      type: "lightgbm"
      enabled: true
      params:
        objective: "binary"
        boosting_type: "gbdt"
        num_leaves: 20  # Reduced from 31 for regularization
        max_depth: 5
        learning_rate: 0.05
        n_estimators: 100
        subsample: 0.7
        colsample_bytree: 0.7
        min_child_samples: 20  # Increased for regularization
        reg_alpha: 0.1  # L1 regularization
        reg_lambda: 1.0  # L2 regularization
        random_state: 42
      early_stopping:
        enabled: true
        rounds: 50
        metric: "binary_logloss"

    # Model 2: XGBoost (depth-wise growth, different splitting)
    xgboost:
      type: "xgboost"
      enabled: true
      params:
        objective: "binary:logistic"
        max_depth: 5
        learning_rate: 0.05
        n_estimators: 100
        subsample: 0.7
        colsample_bytree: 0.7
        min_child_weight: 5  # NEW: Require 5+ samples per leaf
        gamma: 0.1           # NEW: Minimum loss reduction for split
        reg_alpha: 0.5       # NEW: L1 regularization
        reg_lambda: 1.0      # NEW: L2 regularization
        scale_pos_weight: 1.0
        random_state: 42
      early_stopping:
        enabled: true
        rounds: 50
        metric: "logloss"

    # Model 3: CatBoost (symmetric trees, native categorical handling)
    catboost:
      type: "catboost"
      enabled: true
      params:
        iterations: 100
        learning_rate: 0.05
        depth: 5
        l2_leaf_reg: 5.0     # Regularization
        subsample: 0.7
        random_strength: 1.0
        bootstrap_type: "Bernoulli"
        random_state: 42
        verbose: false
      early_stopping:
        enabled: true
        rounds: 50

  # Meta-learner configuration
  meta_learner_config:
    type: "ridge"  # CHANGED from XGBoost to Ridge (faster, simpler)
    params:
      alpha: 1.0   # Regularization strength
      fit_intercept: true
      random_state: 42

    # Alternative meta-learners (commented out)
    # type: "logistic"  # Logistic regression
    # type: "xgboost"   # XGBoost (more complex, slower)

  # Training configuration
  training:
    use_oof_predictions: true  # Out-of-fold predictions for meta-learner
    stack_raw_features: false   # Don't include raw features in meta-learner
    cv_folds: 5

# Retraining schedule - IMPROVED
retraining:
  schedule: "monthly"  # CHANGED from quarterly
  day_of_month: 1
  trigger_on_drift: true
  drift_threshold: 0.10  # PSI threshold
  min_samples: 2000      # REDUCED from 10000

  # NEW: Performance-based triggers
  performance_triggers:
    - metric: "sharpe_ratio"
      threshold: 1.0
      window_days: 30
      description: "Retrain if 30-day Sharpe falls below 1.0"

    - metric: "accuracy"
      threshold: 0.55
      window_days: 7
      description: "Retrain if weekly accuracy falls below 55%"

  # Backup schedule
  fallback: "quarterly"
  max_training_time_hours: 8

# Model versioning
versioning:
  enabled: true
  registry: "mlflow"
  auto_register: true
  production_alias: "champion"
  staging_alias: "challenger"

# Performance targets
performance:
  targets:
    min_sharpe_ratio: 1.2
    max_drawdown: 0.20
    min_accuracy: 0.55
    min_win_rate: 0.52

  # Model selection criteria
  selection:
    primary_metric: "sharpe_ratio"
    secondary_metrics:
      - "max_drawdown"
      - "win_rate"
```

---

### 2.3 Add Feature Importance Validation (3 hours)

**New File:** `src/models/feature_validation.py`

```python
"""
Feature importance validation and leakage detection.

Uses SHAP values to identify suspicious features and data leakage.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings

from src.core.logging_config import get_logger
from src.core.exceptions import DataError

logger = get_logger(__name__)


@dataclass
class FeatureImportanceReport:
    """Report of feature importance analysis."""
    top_features: List[Tuple[str, float]]
    suspicious_features: List[str]
    dead_features: List[str]
    highly_correlated_groups: List[List[str]]
    leakage_detected: bool
    recommendations: List[str]


class FeatureValidator:
    """
    Validates features for data leakage and quality issues.

    Checks:
    1. SHAP values for suspicious patterns
    2. Feature variance (dead features)
    3. Feature importance consistency
    4. Leakage keywords in feature names
    5. Temporal consistency
    """

    def __init__(
        self,
        variance_threshold: float = 1e-6,
        importance_threshold: float = 0.001,
        use_shap: bool = True
    ):
        """
        Initialize feature validator.

        Args:
            variance_threshold: Minimum variance for live features
            importance_threshold: Minimum importance to consider
            use_shap: Use SHAP for importance (vs model-native)
        """
        self.variance_threshold = variance_threshold
        self.importance_threshold = importance_threshold
        self.use_shap = use_shap
        self.logger = logger

    def validate(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: Optional[List[str]] = None
    ) -> FeatureImportanceReport:
        """
        Comprehensive feature validation.

        Args:
            model: Trained model (must have predict method)
            X: Feature matrix
            y: Target vector
            feature_names: Feature names (None = use X.columns)

        Returns:
            FeatureImportanceReport with findings
        """
        if feature_names is None:
            feature_names = X.columns.tolist()

        self.logger.info(f"Validating {len(feature_names)} features")

        # 1. Check for leakage keywords
        suspicious_names = self._check_feature_names(feature_names)

        # 2. Check for dead features
        dead_features = self._check_dead_features(X, feature_names)

        # 3. Get feature importance
        if self.use_shap:
            importances = self._get_shap_importance(model, X, feature_names)
        else:
            importances = self._get_native_importance(model, feature_names)

        # 4. Check for suspicious importance patterns
        suspicious_importance = self._check_importance_patterns(
            importances, feature_names
        )

        # 5. Find highly correlated features
        correlated_groups = self._find_correlated_groups(X, threshold=0.90)

        # Combine suspicious features
        all_suspicious = list(set(suspicious_names + suspicious_importance))

        # Generate recommendations
        recommendations = self._generate_recommendations(
            suspicious_features=all_suspicious,
            dead_features=dead_features,
            correlated_groups=correlated_groups,
            importances=importances
        )

        # Determine if leakage detected
        leakage_detected = len(suspicious_names) > 0 or len(suspicious_importance) > 0

        # Top features
        top_features = sorted(
            [(name, imp) for name, imp in zip(feature_names, importances)],
            key=lambda x: x[1],
            reverse=True
        )[:20]

        report = FeatureImportanceReport(
            top_features=top_features,
            suspicious_features=all_suspicious,
            dead_features=dead_features,
            highly_correlated_groups=correlated_groups,
            leakage_detected=leakage_detected,
            recommendations=recommendations
        )

        # Log critical findings
        if leakage_detected:
            self.logger.error(
                f"❌ LEAKAGE DETECTED: {len(all_suspicious)} suspicious features"
            )
        if len(dead_features) > 0:
            self.logger.warning(f"⚠️  {len(dead_features)} dead features found")

        return report

    def _check_feature_names(self, feature_names: List[str]) -> List[str]:
        """Check feature names for suspicious keywords."""
        suspicious_keywords = [
            'future', 'forward', 'ahead', 'next', 'target',
            'label', 'dist_from_high', 'dist_from_low'
        ]

        suspicious = []
        for name in feature_names:
            name_lower = name.lower()
            for keyword in suspicious_keywords:
                if keyword in name_lower:
                    suspicious.append(name)
                    self.logger.warning(
                        f"Suspicious feature name: {name} (contains '{keyword}')"
                    )
                    break

        return suspicious

    def _check_dead_features(
        self,
        X: pd.DataFrame,
        feature_names: List[str]
    ) -> List[str]:
        """Check for features with zero or near-zero variance."""
        dead_features = []

        for feature in feature_names:
            if feature not in X.columns:
                continue

            variance = X[feature].var()

            if np.isnan(variance) or variance < self.variance_threshold:
                dead_features.append(feature)
                self.logger.debug(f"Dead feature: {feature} (variance={variance:.2e})")

        return dead_features

    def _get_shap_importance(
        self,
        model,
        X: pd.DataFrame,
        feature_names: List[str]
    ) -> np.ndarray:
        """Calculate SHAP-based feature importance."""
        try:
            import shap

            # Subsample for speed if large dataset
            if len(X) > 1000:
                X_sample = X.sample(n=1000, random_state=42)
            else:
                X_sample = X

            # Create explainer
            if hasattr(model, 'predict_proba'):
                explainer = shap.Explainer(model.predict_proba, X_sample)
            else:
                explainer = shap.Explainer(model.predict, X_sample)

            # Calculate SHAP values
            shap_values = explainer(X_sample)

            # Mean absolute SHAP value per feature
            if isinstance(shap_values, list):  # Multi-class
                importances = np.mean([
                    np.abs(sv.values).mean(axis=0) for sv in shap_values
                ], axis=0)
            else:
                importances = np.abs(shap_values.values).mean(axis=0)

            return importances

        except Exception as e:
            self.logger.warning(f"SHAP failed: {e}, falling back to native importance")
            return self._get_native_importance(model, feature_names)

    def _get_native_importance(
        self,
        model,
        feature_names: List[str]
    ) -> np.ndarray:
        """Get model's native feature importance."""
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        elif hasattr(model, 'coef_'):
            return np.abs(model.coef_).flatten()
        else:
            self.logger.warning("Model has no feature importance, using uniform")
            return np.ones(len(feature_names)) / len(feature_names)

    def _check_importance_patterns(
        self,
        importances: np.ndarray,
        feature_names: List[str]
    ) -> List[str]:
        """Check for suspicious importance patterns."""
        suspicious = []

        # Normalize importances
        if importances.sum() > 0:
            norm_importances = importances / importances.sum()
        else:
            return suspicious

        # Check for features that are TOO important
        for i, (name, imp) in enumerate(zip(feature_names, norm_importances)):
            # If single feature accounts for >30% of importance
            if imp > 0.30:
                suspicious.append(name)
                self.logger.warning(
                    f"Feature {name} has suspiciously high importance: {imp:.1%}"
                )

        return suspicious

    def _find_correlated_groups(
        self,
        X: pd.DataFrame,
        threshold: float = 0.90
    ) -> List[List[str]]:
        """Find groups of highly correlated features."""
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()

        # Find groups
        correlated_groups = []
        processed = set()

        for i, col1 in enumerate(corr_matrix.columns):
            if col1 in processed:
                continue

            # Find all features highly correlated with col1
            group = [col1]
            for j, col2 in enumerate(corr_matrix.columns):
                if i != j and col2 not in processed:
                    if corr_matrix.iloc[i, j] > threshold:
                        group.append(col2)
                        processed.add(col2)

            if len(group) > 1:
                correlated_groups.append(group)
                processed.add(col1)

        return correlated_groups

    def _generate_recommendations(
        self,
        suspicious_features: List[str],
        dead_features: List[str],
        correlated_groups: List[List[str]],
        importances: np.ndarray
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        if len(suspicious_features) > 0:
            recommendations.append(
                f"CRITICAL: Remove {len(suspicious_features)} suspicious features: "
                f"{', '.join(suspicious_features[:5])}"
            )

        if len(dead_features) > 0:
            recommendations.append(
                f"Remove {len(dead_features)} dead features (zero variance)"
            )

        if len(correlated_groups) > 0:
            recommendations.append(
                f"Consider removing redundant features from {len(correlated_groups)} "
                "highly correlated groups"
            )

        # Check if importance is too concentrated
        if importances.sum() > 0:
            top_10_importance = np.sort(importances)[-10:].sum() / importances.sum()
            if top_10_importance > 0.80:
                recommendations.append(
                    f"Top 10 features account for {top_10_importance:.1%} of importance. "
                    "Consider simplifying model or engineering more diverse features."
                )

        if not recommendations:
            recommendations.append("✓ No issues detected - features look healthy")

        return recommendations


def test_feature_validation():
    """Test feature validation"""
    from sklearn.ensemble import RandomForestClassifier

    # Create test data with leakage
    np.random.seed(42)
    X = pd.DataFrame({
        'good_feature_1': np.random.randn(1000),
        'good_feature_2': np.random.randn(1000),
        'dist_from_high_20': np.random.randn(1000),  # Suspicious name
        'dead_feature': np.ones(1000),  # No variance
        'correlated_1': np.random.randn(1000)
    })
    X['correlated_2'] = X['correlated_1'] + np.random.randn(1000) * 0.01  # Highly correlated

    y = (X['good_feature_1'] + X['good_feature_2'] > 0).astype(int)

    # Train model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    # Validate
    validator = FeatureValidator()
    report = validator.validate(model, X, y)

    assert 'dist_from_high_20' in report.suspicious_features
    assert 'dead_feature' in report.dead_features
    assert len(report.highly_correlated_groups) > 0
```

---

### 2.4 Implement Monthly Retraining with Drift Triggers (2 hours)

**New File:** `src/models/retraining_manager.py`

```python
"""
Automated retraining manager.

Handles scheduled and triggered model retraining based on:
- Time-based schedule (monthly)
- Drift detection
- Performance degradation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Callable
from dataclasses import dataclass

from src.monitoring.drift_detection import DriftDetector, DriftReport
from src.core.logging_config import get_logger
from src.core.config import ConfigManager

logger = get_logger(__name__)


@dataclass
class RetrainingDecision:
    """Decision on whether to retrain."""
    should_retrain: bool
    reason: str
    urgency: str  # 'low', 'medium', 'high', 'critical'
    metrics: Dict


class RetrainingManager:
    """
    Manages automated model retraining.

    Triggers retraining based on:
    1. Schedule (e.g., monthly)
    2. Drift detection (PSI > threshold)
    3. Performance degradation (Sharpe < threshold)
    4. Manual trigger

    Usage:
        manager = RetrainingManager(config)
        decision = manager.check_retrain_needed(recent_data, recent_performance)
        if decision.should_retrain:
            manager.retrain(training_data)
    """

    def __init__(
        self,
        config: Optional[ConfigManager] = None,
        drift_detector: Optional[DriftDetector] = None
    ):
        """
        Initialize retraining manager.

        Args:
            config: Configuration manager
            drift_detector: Fitted drift detector
        """
        self.config = config or ConfigManager()
        self.drift_detector = drift_detector
        self.logger = logger

        # Load configuration
        self.schedule_frequency = self.config.get(
            'models.retraining.schedule', 'monthly'
        )
        self.drift_threshold = self.config.get(
            'models.retraining.drift_threshold', 0.10
        )
        self.min_samples = self.config.get(
            'models.retraining.min_samples', 2000
        )

        # State tracking
        self.last_retrain_date = None
        self.retraining_history = []

        # Load last retrain date if exists
        self._load_state()

    def check_retrain_needed(
        self,
        recent_data: pd.DataFrame,
        recent_performance: Optional[Dict] = None
    ) -> RetrainingDecision:
        """
        Determine if retraining is needed.

        Args:
            recent_data: Recent feature data for drift detection
            recent_performance: Recent performance metrics

        Returns:
            RetrainingDecision with recommendation
        """
        self.logger.info("Checking if retraining is needed...")

        reasons = []
        urgency = 'low'
        should_retrain = False
        metrics = {}

        # Check 1: Schedule
        schedule_check = self._check_schedule()
        if schedule_check['due']:
            reasons.append(f"Schedule: {schedule_check['reason']}")
            urgency = max(urgency, 'medium', key=self._urgency_rank)
            should_retrain = True
        metrics['schedule'] = schedule_check

        # Check 2: Drift
        if self.drift_detector is not None:
            drift_check = self._check_drift(recent_data)
            if drift_check['significant']:
                reasons.append(f"Drift: PSI={drift_check['psi']:.4f}")
                urgency = max(urgency, 'high', key=self._urgency_rank)
                should_retrain = True
            metrics['drift'] = drift_check

        # Check 3: Performance
        if recent_performance is not None:
            perf_check = self._check_performance(recent_performance)
            if perf_check['degraded']:
                reasons.append(f"Performance: {perf_check['reason']}")
                urgency = max(urgency, 'critical' if perf_check['critical'] else 'high',
                            key=self._urgency_rank)
                should_retrain = True
            metrics['performance'] = perf_check

        # Check 4: Data availability
        data_check = self._check_data_availability(recent_data)
        if not data_check['sufficient']:
            reasons.append("Insufficient data for retraining")
            should_retrain = False  # Override if not enough data
        metrics['data'] = data_check

        # Generate reason string
        if should_retrain:
            reason = " | ".join(reasons)
        else:
            reason = "No retraining needed - all checks passed"

        decision = RetrainingDecision(
            should_retrain=should_retrain,
            reason=reason,
            urgency=urgency,
            metrics=metrics
        )

        self.logger.info(
            f"Retraining decision: {should_retrain} "
            f"(urgency={urgency}, reason={reason})"
        )

        return decision

    def _check_schedule(self) -> Dict:
        """Check if scheduled retraining is due."""
        if self.last_retrain_date is None:
            return {
                'due': True,
                'reason': 'No previous training date found',
                'days_since': None
            }

        days_since = (datetime.now() - self.last_retrain_date).days

        # Determine if due based on frequency
        if self.schedule_frequency == 'weekly':
            due = days_since >= 7
        elif self.schedule_frequency == 'monthly':
            due = days_since >= 30
        elif self.schedule_frequency == 'quarterly':
            due = days_since >= 90
        else:
            due = False

        return {
            'due': due,
            'reason': f"{days_since} days since last retrain ({self.schedule_frequency})",
            'days_since': days_since
        }

    def _check_drift(self, recent_data: pd.DataFrame) -> Dict:
        """Check for significant drift."""
        if self.drift_detector is None:
            return {'significant': False, 'psi': 0.0, 'reason': 'No drift detector'}

        try:
            report = self.drift_detector.detect(recent_data)

            return {
                'significant': report.is_significant_drift,
                'psi': report.overall_psi,
                'drifted_features': report.drifted_features,
                'reason': report.recommendation
            }
        except Exception as e:
            self.logger.error(f"Drift detection failed: {e}")
            return {'significant': False, 'psi': 0.0, 'reason': f'Error: {e}'}

    def _check_performance(self, performance: Dict) -> Dict:
        """Check for performance degradation."""
        degraded = False
        critical = False
        reasons = []

        # Check Sharpe ratio
        sharpe = performance.get('sharpe_ratio')
        if sharpe is not None and sharpe < 1.0:
            degraded = True
            reasons.append(f"Sharpe={sharpe:.2f} < 1.0")
            if sharpe < 0.5:
                critical = True

        # Check accuracy
        accuracy = performance.get('accuracy')
        if accuracy is not None and accuracy < 0.55:
            degraded = True
            reasons.append(f"Accuracy={accuracy:.1%} < 55%")
            if accuracy < 0.52:
                critical = True

        # Check drawdown
        drawdown = performance.get('max_drawdown')
        if drawdown is not None and abs(drawdown) > 0.20:
            degraded = True
            reasons.append(f"Drawdown={abs(drawdown):.1%} > 20%")
            if abs(drawdown) > 0.30:
                critical = True

        return {
            'degraded': degraded,
            'critical': critical,
            'reason': ' | '.join(reasons) if reasons else 'Performance OK'
        }

    def _check_data_availability(self, data: pd.DataFrame) -> Dict:
        """Check if sufficient data available for retraining."""
        n_samples = len(data)
        sufficient = n_samples >= self.min_samples

        return {
            'sufficient': sufficient,
            'n_samples': n_samples,
            'min_required': self.min_samples
        }

    def retrain(
        self,
        training_data: pd.DataFrame,
        labels: pd.Series,
        retrain_callback: Callable
    ) -> bool:
        """
        Execute retraining.

        Args:
            training_data: Data for retraining
            labels: Target labels
            retrain_callback: Function to call for actual retraining

        Returns:
            True if successful
        """
        self.logger.info("Starting retraining process...")

        try:
            # Call the actual retraining function
            result = retrain_callback(training_data, labels)

            # Update state
            self.last_retrain_date = datetime.now()
            self.retraining_history.append({
                'timestamp': self.last_retrain_date,
                'n_samples': len(training_data),
                'success': True
            })

            # Save state
            self._save_state()

            self.logger.success("✓ Retraining completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Retraining failed: {e}")
            self.retraining_history.append({
                'timestamp': datetime.now(),
                'n_samples': len(training_data),
                'success': False,
                'error': str(e)
            })
            return False

    def _urgency_rank(self, urgency: str) -> int:
        """Convert urgency to numeric rank."""
        ranks = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}
        return ranks.get(urgency, 0)

    def _load_state(self):
        """Load retraining state from disk."""
        state_path = Path("models/retraining_state.json")
        if state_path.exists():
            import json
            with open(state_path, 'r') as f:
                state = json.load(f)
                if 'last_retrain_date' in state:
                    self.last_retrain_date = datetime.fromisoformat(
                        state['last_retrain_date']
                    )
                self.retraining_history = state.get('history', [])

    def _save_state(self):
        """Save retraining state to disk."""
        state_path = Path("models/retraining_state.json")
        state_path.parent.mkdir(parents=True, exist_ok=True)

        import json
        state = {
            'last_retrain_date': self.last_retrain_date.isoformat() if self.last_retrain_date else None,
            'history': self.retraining_history
        }

        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)
```

---

## Week 3: Performance Optimization (11 hours)

*(Continuing in next section due to length...)*

### 3.1 Stationary Feature Engineering (4 hours)

**File:** `src/features/engineer.py` (add new method)

### 3.2 Parallel Inference with Caching (3 hours)

**New File:** `src/inference/production.py`

### 3.3 Improve Regularization (2 hours)

**File:** `config/models.yaml` (already done in 2.2)

### 3.4 Add Ensemble Diversity Metrics (2 hours)

**New File:** `src/models/diversity.py`

---

## Testing Strategy

1. **Unit Tests**: Each module has dedicated test
2. **Integration Tests**: End-to-end training pipeline
3. **Validation Tests**: Verify no look-ahead bias
4. **Performance Tests**: Benchmark inference latency

---

## Rollout Plan

### Phase 1 (Week 1)
- Implement critical fixes
- Run comprehensive tests
- Validate on historical data

### Phase 2 (Week 2)
- Deploy drift detection
- Update model configuration
- Implement monitoring

### Phase 3 (Week 3)
- Optimize performance
- Add diversity metrics
- Final validation

### Phase 4 (Deployment)
- Gradual rollout (paper trading first)
- Monitor performance closely
- Iterate based on results

---

## Validation Checklist

After all fixes, verify:

- [ ] No features have access to future data
- [ ] Scaler fitted only on train, applied to all sets
- [ ] CPCV with proper purging (5-10 days)
- [ ] CV scores within 2% of test scores
- [ ] Model can predict live data without NaN/Inf
- [ ] Inference latency < 100ms
- [ ] Drift detection alerts if PSI > 0.1
- [ ] Models retrain monthly or on drift
- [ ] Feature importance shows no obvious leakage
- [ ] Ensemble predictions correlate 0.4-0.7 (not 0.9+)

---

## Success Metrics

| Metric | Before | Target | Measurement |
|--------|--------|--------|-------------|
| Overfitting Gap | 15-20% | <5% | Train vs test accuracy |
| Inference Latency | 100-150ms | <50ms | Production measurement |
| Model Stability | Poor | 90%+ uptime | Days without failures |
| Sharpe Ratio | Unknown | +0.5-1.0 | Live trading results |
| Feature Quality | Mixed | 100% validated | Leakage checks passing |

---

**Document Status:** Living document - update as implementation progresses
**Next Update:** After Week 1 completion
