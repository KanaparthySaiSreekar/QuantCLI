"""
Ensemble model implementation.

Combines multiple base models using stacking or averaging:
- XGBoost
- LightGBM
- CatBoost
- LSTM (PyTorch)
- Meta-learner for combining predictions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import warnings

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score

from .base import BaseModel
from src.core.exceptions import ModelError
from src.core.logging_config import get_logger

logger = get_logger(__name__)


class LSTMModel(nn.Module):
    """
    LSTM model for time series prediction.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1
    ):
        """
        Initialize LSTM model.

        Args:
            input_size: Number of input features
            hidden_size: LSTM hidden layer size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            output_size: Number of output units
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Forward pass."""
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)

        # Take last time step
        out = lstm_out[:, -1, :]

        out = self.dropout(out)
        out = self.fc(out)

        return out


class EnsembleModel(BaseModel):
    """
    Ensemble model combining multiple base learners.

    Supports:
    - XGBoost
    - LightGBM
    - CatBoost
    - LSTM
    - Stacking with meta-learner
    """

    def __init__(
        self,
        model_name: str = "ensemble",
        config: Optional[Dict[str, Any]] = None,
        task: str = "classification",
        combine_method: str = "stack"
    ):
        """
        Initialize ensemble model.

        Args:
            model_name: Model identifier
            config: Configuration dictionary
            task: 'classification' or 'regression'
            combine_method: 'stack' (meta-learner) or 'average'
        """
        super().__init__(model_name, config)

        self.task = task
        self.combine_method = combine_method
        self.base_models = {}
        self.meta_learner = None

        # Check available libraries
        self.available_models = self._check_available_models()

        self.logger.info(
            f"Initialized {model_name} ensemble with {len(self.available_models)} "
            f"available base models: {list(self.available_models.keys())}"
        )

    def _check_available_models(self) -> Dict[str, bool]:
        """Check which model libraries are available."""
        return {
            'xgboost': XGBOOST_AVAILABLE,
            'lightgbm': LIGHTGBM_AVAILABLE,
            'catboost': CATBOOST_AVAILABLE,
            'lstm': PYTORCH_AVAILABLE
        }

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train ensemble model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            **kwargs: Additional arguments

        Returns:
            Dictionary with training metrics
        """
        self.feature_names = list(X_train.columns)
        metrics = {}

        # Train base models
        self.logger.info("Training base models...")

        if self.available_models.get('xgboost'):
            metrics['xgboost'] = self._train_xgboost(
                X_train, y_train, X_val, y_val
            )

        if self.available_models.get('lightgbm'):
            metrics['lightgbm'] = self._train_lightgbm(
                X_train, y_train, X_val, y_val
            )

        if self.available_models.get('catboost'):
            metrics['catboost'] = self._train_catboost(
                X_train, y_train, X_val, y_val
            )

        # Train meta-learner if using stacking
        if self.combine_method == 'stack' and len(self.base_models) > 1:
            self.logger.info("Training meta-learner...")
            metrics['meta_learner'] = self._train_meta_learner(
                X_train, y_train, X_val, y_val
            )

        self.is_trained = True
        self.training_metadata = {
            'n_samples': len(X_train),
            'n_features': len(self.feature_names),
            'base_models': list(self.base_models.keys()),
            'task': self.task,
            'combine_method': self.combine_method
        }

        self.logger.info(
            f"Ensemble training complete. Base models: {len(self.base_models)}"
        )

        return metrics

    def _train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series]
    ) -> Dict[str, float]:
        """Train XGBoost model."""
        try:
            params = self.config.get('xgboost', {
                'max_depth': 6,
                'learning_rate': 0.01,
                'n_estimators': 500,
                'objective': 'binary:logistic' if self.task == 'classification' else 'reg:squarederror',
                'random_state': 42,
                'n_jobs': -1
            })

            if self.task == 'classification':
                model = xgb.XGBClassifier(**params)
            else:
                model = xgb.XGBRegressor(**params)

            # Prepare validation set for early stopping
            eval_set = [(X_train, y_train)]
            if X_val is not None and y_val is not None:
                eval_set.append((X_val, y_val))

            model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )

            self.base_models['xgboost'] = model
            self.logger.info("XGBoost training complete")

            # Get best score
            best_score = model.best_score if hasattr(model, 'best_score') else 0.0

            return {'best_score': best_score}

        except Exception as e:
            self.logger.error(f"XGBoost training failed: {e}")
            return {}

    def _train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series]
    ) -> Dict[str, float]:
        """Train LightGBM model."""
        try:
            params = self.config.get('lightgbm', {
                'max_depth': 6,
                'learning_rate': 0.01,
                'n_estimators': 500,
                'objective': 'binary' if self.task == 'classification' else 'regression',
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            })

            if self.task == 'classification':
                model = lgb.LGBMClassifier(**params)
            else:
                model = lgb.LGBMRegressor(**params)

            # Prepare validation set
            eval_set = [(X_train, y_train)]
            if X_val is not None and y_val is not None:
                eval_set.append((X_val, y_val))

            model.fit(
                X_train, y_train,
                eval_set=eval_set if X_val is not None else None,
                callbacks=[lgb.log_evaluation(0)]  # Suppress output
            )

            self.base_models['lightgbm'] = model
            self.logger.info("LightGBM training complete")

            best_score = model.best_score_.get('valid_1', {}).get('binary_logloss', 0.0) if hasattr(model, 'best_score_') else 0.0

            return {'best_score': best_score}

        except Exception as e:
            self.logger.error(f"LightGBM training failed: {e}")
            return {}

    def _train_catboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series]
    ) -> Dict[str, float]:
        """Train CatBoost model."""
        try:
            params = self.config.get('catboost', {
                'iterations': 500,
                'learning_rate': 0.01,
                'depth': 6,
                'random_state': 42,
                'verbose': False
            })

            if self.task == 'classification':
                model = CatBoostClassifier(**params)
            else:
                model = CatBoostRegressor(**params)

            # Prepare validation pool
            if X_val is not None and y_val is not None:
                model.fit(
                    X_train, y_train,
                    eval_set=(X_val, y_val),
                    verbose=False
                )
            else:
                model.fit(X_train, y_train, verbose=False)

            self.base_models['catboost'] = model
            self.logger.info("CatBoost training complete")

            best_score = model.get_best_score().get('validation', {}).get('Logloss', 0.0) if X_val is not None else 0.0

            return {'best_score': best_score}

        except Exception as e:
            self.logger.error(f"CatBoost training failed: {e}")
            return {}

    def _train_meta_learner(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series]
    ) -> Dict[str, float]:
        """Train meta-learner for stacking."""
        try:
            # Generate predictions from base models
            base_predictions_train = self._get_base_predictions(X_train)

            if base_predictions_train.shape[1] == 0:
                self.logger.warning("No base model predictions available")
                return {}

            # Train meta-learner
            if self.task == 'classification':
                meta_model = LogisticRegression(random_state=42, max_iter=1000)
            else:
                meta_model = Ridge(random_state=42)

            meta_model.fit(base_predictions_train, y_train)

            self.meta_learner = meta_model

            # Evaluate on validation set if available
            score = 0.0
            if X_val is not None and y_val is not None:
                base_predictions_val = self._get_base_predictions(X_val)
                score = meta_model.score(base_predictions_val, y_val)

            self.logger.info(f"Meta-learner training complete. Score: {score:.4f}")

            return {'score': score}

        except Exception as e:
            self.logger.error(f"Meta-learner training failed: {e}")
            return {}

    def _get_base_predictions(self, X: pd.DataFrame) -> np.ndarray:
        """Get predictions from all base models."""
        predictions = []

        for model_name, model in self.base_models.items():
            try:
                if self.task == 'classification' and hasattr(model, 'predict_proba'):
                    # Use probabilities for classification
                    pred = model.predict_proba(X)[:, 1]
                else:
                    pred = model.predict(X)

                predictions.append(pred)

            except Exception as e:
                self.logger.warning(f"Failed to get predictions from {model_name}: {e}")

        if not predictions:
            return np.array([]).reshape(len(X), 0)

        return np.column_stack(predictions)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using ensemble.

        Args:
            X: Features to predict on

        Returns:
            Array of predictions
        """
        X = self._validate_input(X)

        if self.combine_method == 'stack' and self.meta_learner is not None:
            # Use stacking
            base_predictions = self._get_base_predictions(X)
            predictions = self.meta_learner.predict(base_predictions)

        else:
            # Use averaging
            base_predictions = self._get_base_predictions(X)

            if base_predictions.shape[1] == 0:
                raise ModelError("No base models available for prediction")

            predictions = base_predictions.mean(axis=1)

        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Features to predict on

        Returns:
            Array of probabilities
        """
        if self.task != 'classification':
            raise ModelError("predict_proba only available for classification")

        X = self._validate_input(X)

        if self.combine_method == 'stack' and self.meta_learner is not None:
            base_predictions = self._get_base_predictions(X)
            proba = self.meta_learner.predict_proba(base_predictions)

        else:
            # Average probabilities from base models
            all_proba = []

            for model_name, model in self.base_models.items():
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)
                    all_proba.append(proba)

            if not all_proba:
                raise ModelError("No models support probability predictions")

            proba = np.mean(all_proba, axis=0)

        return proba

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get aggregated feature importance from all base models."""
        if not self.is_trained:
            raise ModelError("Model must be trained first")

        all_importance = []

        for model_name, model in self.base_models.items():
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': model.feature_importances_,
                    'model': model_name
                })
                all_importance.append(importance_df)

        if not all_importance:
            self.logger.warning("No models support feature importance")
            return None

        # Combine and average
        combined = pd.concat(all_importance)
        avg_importance = combined.groupby('feature')['importance'].mean().reset_index()
        avg_importance = avg_importance.sort_values('importance', ascending=False)

        return avg_importance
