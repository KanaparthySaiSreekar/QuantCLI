"""
Base model interface for all ML models.

Provides common interface for training, prediction, and serialization.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from datetime import datetime

from src.core.logging_config import get_logger
from src.core.exceptions import ModelError

logger = get_logger(__name__)


class BaseModel(ABC):
    """
    Abstract base class for all ML models.

    Provides consistent interface for:
    - Training
    - Prediction
    - Model persistence
    - Evaluation
    """

    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize base model.

        Args:
            model_name: Name identifier for the model
            config: Model configuration dictionary
        """
        self.model_name = model_name
        self.config = config or {}
        self.model = None
        self.is_trained = False
        self.feature_names = []
        self.training_metadata = {}
        self.logger = logger

    @abstractmethod
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            **kwargs: Additional training arguments

        Returns:
            Dictionary with training metrics

        Raises:
            ModelError: If training fails
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features to predict on

        Returns:
            Array of predictions

        Raises:
            ModelError: If model not trained or prediction fails
        """
        pass

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities (for classification models).

        Args:
            X: Features to predict on

        Returns:
            Array of class probabilities

        Raises:
            ModelError: If not supported by model
        """
        raise NotImplementedError(
            f"{self.model_name} does not support probability predictions"
        )

    def save(self, path: Path) -> None:
        """
        Save model to disk.

        Args:
            path: Path to save model file

        Raises:
            ModelError: If save fails
        """
        if not self.is_trained:
            raise ModelError(f"Cannot save untrained model: {self.model_name}")

        try:
            path.parent.mkdir(parents=True, exist_ok=True)

            model_data = {
                'model': self.model,
                'model_name': self.model_name,
                'config': self.config,
                'feature_names': self.feature_names,
                'training_metadata': self.training_metadata,
                'saved_at': datetime.now().isoformat()
            }

            joblib.dump(model_data, path)
            self.logger.info(f"Saved {self.model_name} to {path}")

        except Exception as e:
            raise ModelError(f"Failed to save model: {e}") from e

    def load(self, path: Path) -> None:
        """
        Load model from disk.

        Args:
            path: Path to model file

        Raises:
            ModelError: If load fails
        """
        if not path.exists():
            raise ModelError(f"Model file not found: {path}")

        try:
            model_data = joblib.load(path)

            self.model = model_data['model']
            self.model_name = model_data['model_name']
            self.config = model_data['config']
            self.feature_names = model_data['feature_names']
            self.training_metadata = model_data.get('training_metadata', {})
            self.is_trained = True

            self.logger.info(f"Loaded {self.model_name} from {path}")

        except Exception as e:
            raise ModelError(f"Failed to load model: {e}") from e

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance scores.

        Returns:
            DataFrame with feature names and importance scores,
            or None if not supported
        """
        if not self.is_trained:
            raise ModelError("Model must be trained first")

        if not hasattr(self.model, 'feature_importances_'):
            self.logger.warning(f"{self.model_name} does not support feature importance")
            return None

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        return importance_df

    def _validate_input(self, X: pd.DataFrame) -> None:
        """Validate input features."""
        if not self.is_trained:
            raise ModelError(f"Model {self.model_name} is not trained")

        if X.empty:
            raise ModelError("Input features cannot be empty")

        # Check for feature mismatch
        if self.feature_names:
            missing_features = set(self.feature_names) - set(X.columns)
            if missing_features:
                raise ModelError(
                    f"Missing features: {missing_features}"
                )

            # Ensure correct order
            X = X[self.feature_names]

        return X

    def __repr__(self) -> str:
        """String representation."""
        status = "trained" if self.is_trained else "untrained"
        return f"{self.__class__.__name__}(name={self.model_name}, status={status})"
