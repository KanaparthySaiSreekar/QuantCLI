"""
Model training orchestration.

Handles:
- Data splitting
- Cross-validation
- Model training workflow
- Hyperparameter tuning
- Training pipeline
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from .base import BaseModel
from .ensemble import EnsembleModel
from src.core.logging_config import get_logger
from src.core.exceptions import ModelError, ValidationError

logger = get_logger(__name__)


class ModelTrainer:
    """
    Orchestrates model training workflow.

    Handles:
    - Data preprocessing
    - Train/validation/test splits
    - Model training
    - Model evaluation
    - Model persistence
    """

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
        """
        Initialize trainer.

        Args:
            model: Model instance to train
            task: 'classification' or 'regression'
            validation_method: 'holdout', 'timeseries', or 'cpcv'
            test_size: Fraction of data for test set
            val_size: Fraction of training data for validation
            n_splits: Number of CV splits for time series methods
            random_state: Random seed
            scale_features: Whether to scale features
        """
        self.model = model
        self.task = task
        self.validation_method = validation_method
        self.test_size = test_size
        self.val_size = val_size
        self.n_splits = n_splits
        self.random_state = random_state
        self.scale_features = scale_features

        # FIXED: Fit scaler ONCE, not per fold
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
        """Train with time-aware holdout validation (NO random shuffle for time series)."""
        # FIXED: Use time-aware split (no shuffle) instead of random split
        n = len(X)
        train_end = int(n * (1 - self.test_size - self.val_size))
        val_end = int(n * (1 - self.test_size))

        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]

        if self.val_size > 0:
            X_val = X.iloc[train_end:val_end]
            y_val = y.iloc[train_end:val_end]
        else:
            X_val, y_val = None, None

        X_test = X.iloc[val_end:]
        y_test = y.iloc[val_end:]

        self.logger.info(
            f"Time-aware split: train={len(X_train)}, "
            f"val={len(X_val) if X_val is not None else 0}, "
            f"test={len(X_test)}"
        )

        # FIXED: Fit scaler ONLY on training data (once)
        if self.scale_features and not self.scaler_fitted:
            self.scaler.fit(X_train)
            self.scaler_fitted = True
            self.logger.info("Scaler fitted on training data only")

        # Transform all sets with same scaler (no refitting)
        if self.scale_features:
            X_train = self._scale_transform(X_train)
            if X_val is not None:
                X_val = self._scale_transform(X_val)
            X_test = self._scale_transform(X_test)

        # Train model
        self.logger.info("Training model...")
        train_metrics = self.model.train(
            X_train, y_train,
            X_val, y_val
        )

        # Evaluate on test set
        self.logger.info("Evaluating on test set...")
        test_predictions = self.model.predict(X_test)

        # Calculate metrics
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            mean_squared_error, mean_absolute_error, r2_score
        )

        if self.task == 'classification':
            test_metrics = {
                'accuracy': accuracy_score(y_test, test_predictions.round()),
                'precision': precision_score(y_test, test_predictions.round(), average='weighted', zero_division=0),
                'recall': recall_score(y_test, test_predictions.round(), average='weighted', zero_division=0),
                'f1': f1_score(y_test, test_predictions.round(), average='weighted', zero_division=0)
            }
        else:
            test_metrics = {
                'mse': mean_squared_error(y_test, test_predictions),
                'rmse': np.sqrt(mean_squared_error(y_test, test_predictions)),
                'mae': mean_absolute_error(y_test, test_predictions),
                'r2': r2_score(y_test, test_predictions)
            }

        self.logger.info(f"Test metrics: {test_metrics}")

        # Store history
        self.training_history = {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'n_train': len(X_train),
            'n_val': len(X_val) if X_val is not None else 0,
            'n_test': len(X_test),
            'trained_at': datetime.now().isoformat()
        }

        return {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'training_history': self.training_history
        }

    def _train_time_series_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, Any]:
        """Train with time series cross-validation."""
        self.logger.info(f"Using time series CV with {self.n_splits} splits")

        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        # FIXED: Fit scaler ONCE on initial training data
        if self.scale_features and not self.scaler_fitted:
            # Use first 80% for scaler fitting
            train_size = int(len(X) * 0.8)
            self.scaler.fit(X.iloc[:train_size])
            self.scaler_fitted = True
            self.logger.info("Scaler fitted on initial 80% of data")

        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            self.logger.info(f"Training fold {fold}/{self.n_splits}")

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # FIXED: Transform only (DON'T refit scaler)
            if self.scale_features:
                X_train = self._scale_transform(X_train)
                X_val = self._scale_transform(X_val)

            # Train
            self.model.train(X_train, y_train, X_val, y_val)

            # Evaluate
            predictions = self.model.predict(X_val)
            score = self._calculate_fold_score(y_train, y_val, predictions)

            fold_scores.append(score)
            self.logger.info(f"Fold {fold} score: {score:.4f}")

        avg_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)

        self.logger.info(
            f"CV complete. Average score: {avg_score:.4f} (+/- {std_score:.4f})"
        )

        # Final train on all data (except holdout test set)
        self.logger.info("Training final model on all data...")

        test_size = int(len(X) * self.test_size)
        X_train_full = X.iloc[:-test_size]
        y_train_full = y.iloc[:-test_size]
        X_test = X.iloc[-test_size:]
        y_test = y.iloc[-test_size:]

        # FIXED: Transform only (scaler already fitted)
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

        # FIXED: Fit scaler ONCE on initial training data
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

            # FIXED: Transform only (DON'T refit scaler)
            if self.scale_features:
                X_train = self._scale_transform(X_train)
                X_val = self._scale_transform(X_val)

            # Train
            self.model.train(X_train, y_train, X_val, y_val)

            # Evaluate
            predictions = self.model.predict(X_val)
            score = self._calculate_fold_score(y_train, y_val, predictions)

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
        """
        Transform data using FITTED scaler (no refitting).

        IMPORTANT: This method never refits the scaler. The scaler must be
        fitted once on training data using scaler.fit() directly.
        """
        if self.scaler is None or not self.scaler_fitted:
            return X

        X_scaled = self.scaler.transform(X)
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    def _calculate_fold_score(
        self,
        y_train: pd.Series,
        y_val: pd.Series,
        y_pred: np.ndarray
    ) -> float:
        """Calculate single score for CV fold."""
        from sklearn.metrics import accuracy_score, mean_squared_error

        if self.task == 'classification':
            return accuracy_score(y_val, y_pred.round())
        else:
            return -mean_squared_error(y_val, y_pred)  # Negative MSE

    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict:
        """Calculate comprehensive metrics."""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            mean_squared_error, mean_absolute_error, r2_score
        )

        if self.task == 'classification':
            return {
                'accuracy': accuracy_score(y_true, y_pred.round()),
                'precision': precision_score(
                    y_true, y_pred.round(),
                    average='weighted',
                    zero_division=0
                ),
                'recall': recall_score(
                    y_true, y_pred.round(),
                    average='weighted',
                    zero_division=0
                ),
                'f1': f1_score(
                    y_true, y_pred.round(),
                    average='weighted',
                    zero_division=0
                )
            }
        else:
            return {
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred)
            }

    def _validate_data(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Validate input data."""
        if X.empty or y.empty:
            raise ValidationError("Features and target cannot be empty")

        if len(X) != len(y):
            raise ValidationError(
                f"Feature and target length mismatch: {len(X)} vs {len(y)}"
            )

        if X.isnull().any().any():
            n_nulls = X.isnull().sum().sum()
            raise ValidationError(f"Features contain {n_nulls} null values")

        if y.isnull().any():
            n_nulls = y.isnull().sum()
            raise ValidationError(f"Target contains {n_nulls} null values")

    def save_model(self, path: Path) -> None:
        """
        Save trained model and scaler.

        Args:
            path: Directory to save model
        """
        if not self.model.is_trained:
            raise ModelError("Cannot save untrained model")

        path.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = path / f"{self.model.model_name}.joblib"
        self.model.save(model_path)

        # Save scaler
        if self.scaler is not None:
            import joblib
            scaler_path = path / f"{self.model.model_name}_scaler.joblib"
            joblib.dump(self.scaler, scaler_path)
            self.logger.info(f"Saved scaler to {scaler_path}")

        # Save training history
        import json
        history_path = path / f"{self.model.model_name}_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)

        self.logger.info(f"Saved model artifacts to {path}")

    def load_model(self, path: Path) -> None:
        """
        Load trained model and scaler.

        Args:
            path: Directory containing model artifacts
        """
        # Load model
        model_path = path / f"{self.model.model_name}.joblib"
        self.model.load(model_path)

        # Load scaler if exists
        scaler_path = path / f"{self.model.model_name}_scaler.joblib"
        if scaler_path.exists():
            import joblib
            self.scaler = joblib.load(scaler_path)
            self.logger.info(f"Loaded scaler from {scaler_path}")

        # Load history if exists
        history_path = path / f"{self.model.model_name}_history.json"
        if history_path.exists():
            import json
            with open(history_path, 'r') as f:
                self.training_history = json.load(f)

        self.logger.info(f"Loaded model artifacts from {path}")


class TrainingPipeline:
    """
    Complete training pipeline for multiple models.

    Orchestrates training, evaluation, and selection of best model.
    """

    def __init__(self, models: List[BaseModel], task: str = "classification"):
        """
        Initialize pipeline.

        Args:
            models: List of models to train
            task: Task type
        """
        self.models = models
        self.task = task
        self.trainers = {
            model.model_name: ModelTrainer(model, task)
            for model in models
        }
        self.results = {}
        self.best_model = None

        self.logger = logger

    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        time_series_split: bool = False
    ) -> Dict[str, Any]:
        """
        Run training pipeline for all models.

        Args:
            X: Features
            y: Target
            time_series_split: Use time series CV

        Returns:
            Dictionary with all results
        """
        self.logger.info(f"Starting training pipeline for {len(self.models)} models")

        for model_name, trainer in self.trainers.items():
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Training: {model_name}")
            self.logger.info(f"{'='*60}")

            try:
                result = trainer.train(X, y, time_series_split)
                self.results[model_name] = result

            except Exception as e:
                self.logger.error(f"Training failed for {model_name}: {e}")
                self.results[model_name] = {'error': str(e)}

        # Select best model
        self.best_model = self._select_best_model()

        self.logger.info(f"\nBest model: {self.best_model}")

        return {
            'results': self.results,
            'best_model': self.best_model
        }

    def _select_best_model(self) -> str:
        """Select best model based on test metrics."""
        if not self.results:
            return None

        best_model = None
        best_score = -np.inf

        metric_key = 'accuracy' if self.task == 'classification' else 'r2'

        for model_name, result in self.results.items():
            if 'error' in result:
                continue

            # Get test score
            test_metrics = result.get('test_metrics', {})
            score = test_metrics.get(metric_key, -np.inf)

            if score > best_score:
                best_score = score
                best_model = model_name

        return best_model

    def save_all(self, base_path: Path) -> None:
        """Save all trained models."""
        for model_name, trainer in self.trainers.items():
            model_path = base_path / model_name
            try:
                trainer.save_model(model_path)
            except Exception as e:
                self.logger.error(f"Failed to save {model_name}: {e}")
