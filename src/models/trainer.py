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
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        scale_features: bool = True
    ):
        """
        Initialize trainer.

        Args:
            model: Model instance to train
            task: 'classification' or 'regression'
            test_size: Fraction of data for test set
            val_size: Fraction of training data for validation
            random_state: Random seed
            scale_features: Whether to scale features
        """
        self.model = model
        self.task = task
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.scale_features = scale_features

        self.scaler = StandardScaler() if scale_features else None
        self.training_history = {}

        self.logger = logger

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        time_series_split: bool = False,
        n_splits: int = 5
    ) -> Dict[str, Any]:
        """
        Train model with data splitting.

        Args:
            X: Feature DataFrame
            y: Target Series
            time_series_split: Use time series cross-validation
            n_splits: Number of CV splits for time series

        Returns:
            Dictionary with training results
        """
        self.logger.info(f"Starting training for {self.model.model_name}")
        self.logger.info(f"Dataset: {len(X)} samples, {len(X.columns)} features")

        # Validate inputs
        self._validate_data(X, y)

        # Split data
        if time_series_split:
            return self._train_time_series_cv(X, y, n_splits)
        else:
            return self._train_holdout(X, y)

    def _train_holdout(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train with holdout validation."""
        # Split into train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y if self.task == 'classification' else None
        )

        # Split train into train and val
        if self.val_size > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=self.val_size / (1 - self.test_size),
                random_state=self.random_state,
                stratify=y_temp if self.task == 'classification' else None
            )
        else:
            X_train, y_train = X_temp, y_temp
            X_val, y_val = None, None

        self.logger.info(
            f"Split: train={len(X_train)}, val={len(X_val) if X_val is not None else 0}, "
            f"test={len(X_test)}"
        )

        # Scale features
        if self.scale_features:
            X_train = self._scale_fit_transform(X_train)
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
        y: pd.Series,
        n_splits: int
    ) -> Dict[str, Any]:
        """Train with time series cross-validation."""
        self.logger.info(f"Using time series CV with {n_splits} splits")

        tscv = TimeSeriesSplit(n_splits=n_splits)

        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            self.logger.info(f"Training fold {fold}/{n_splits}")

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Scale
            if self.scale_features:
                X_train = self._scale_fit_transform(X_train)
                X_val = self._scale_transform(X_val)

            # Train
            self.model.train(X_train, y_train, X_val, y_val)

            # Evaluate
            predictions = self.model.predict(X_val)

            from sklearn.metrics import accuracy_score, mean_squared_error

            if self.task == 'classification':
                score = accuracy_score(y_val, predictions.round())
            else:
                score = -mean_squared_error(y_val, predictions)  # Negative MSE

            fold_scores.append(score)
            self.logger.info(f"Fold {fold} score: {score:.4f}")

        avg_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)

        self.logger.info(
            f"CV complete. Average score: {avg_score:.4f} (+/- {std_score:.4f})"
        )

        # Final train on all data
        self.logger.info("Training final model on all data...")

        if self.scale_features:
            X_scaled = self._scale_fit_transform(X)
        else:
            X_scaled = X

        final_metrics = self.model.train(X_scaled, y)

        self.training_history = {
            'cv_scores': fold_scores,
            'avg_cv_score': avg_score,
            'std_cv_score': std_score,
            'final_metrics': final_metrics,
            'n_samples': len(X),
            'trained_at': datetime.now().isoformat()
        }

        return {
            'cv_scores': fold_scores,
            'avg_score': avg_score,
            'std_score': std_score,
            'final_metrics': final_metrics
        }

    def _scale_fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit scaler and transform data."""
        if self.scaler is None:
            return X

        X_scaled = self.scaler.fit_transform(X)
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    def _scale_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted scaler."""
        if self.scaler is None:
            return X

        X_scaled = self.scaler.transform(X)
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

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
