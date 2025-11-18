"""
Model evaluation framework.

Provides comprehensive model evaluation metrics and visualizations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)

from src.core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationResult:
    """
    Results from model evaluation.

    Attributes:
        task: 'classification' or 'regression'
        metrics: Dictionary of metric scores
        predictions: Model predictions
        actuals: Actual values
        metadata: Additional information
    """
    task: str
    metrics: Dict[str, float]
    predictions: np.ndarray
    actuals: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """String representation of results."""
        metrics_str = "\n".join([f"  {k}: {v:.4f}" for k, v in self.metrics.items()])
        return f"""
Evaluation Results ({self.task}):
{metrics_str}
"""


class ModelEvaluator:
    """
    Evaluates model performance with comprehensive metrics.

    Supports both classification and regression tasks.
    """

    def __init__(self, task: str = "classification"):
        """
        Initialize evaluator.

        Args:
            task: 'classification' or 'regression'
        """
        self.task = task
        self.logger = logger

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> EvaluationResult:
        """
        Evaluate predictions against ground truth.

        Args:
            y_true: True labels/values
            y_pred: Predicted labels/values
            y_pred_proba: Predicted probabilities (classification only)

        Returns:
            EvaluationResult object
        """
        if self.task == 'classification':
            metrics = self._evaluate_classification(y_true, y_pred, y_pred_proba)
        else:
            metrics = self._evaluate_regression(y_true, y_pred)

        result = EvaluationResult(
            task=self.task,
            metrics=metrics,
            predictions=y_pred,
            actuals=y_true,
            metadata={
                'n_samples': len(y_true),
                'task': self.task
            }
        )

        self.logger.info(f"Evaluation complete:\n{result}")

        return result

    def _evaluate_classification(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate classification metrics."""
        metrics = {}

        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(
            y_true, y_pred, average='weighted', zero_division=0
        )
        metrics['recall'] = recall_score(
            y_true, y_pred, average='weighted', zero_division=0
        )
        metrics['f1'] = f1_score(
            y_true, y_pred, average='weighted', zero_division=0
        )

        # ROC AUC if probabilities available
        if y_pred_proba is not None:
            try:
                # For binary classification
                if y_pred_proba.ndim == 1 or y_pred_proba.shape[1] == 2:
                    proba = y_pred_proba if y_pred_proba.ndim == 1 else y_pred_proba[:, 1]
                    metrics['roc_auc'] = roc_auc_score(y_true, proba)
                else:
                    # Multi-class
                    metrics['roc_auc'] = roc_auc_score(
                        y_true, y_pred_proba, multi_class='ovr', average='weighted'
                    )
            except Exception as e:
                self.logger.warning(f"Could not calculate ROC AUC: {e}")

        # Confusion matrix stats
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):  # Binary classification
            tn, fp, fn, tp = cm.ravel()
            metrics['true_positives'] = tp
            metrics['true_negatives'] = tn
            metrics['false_positives'] = fp
            metrics['false_negatives'] = fn
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0

        return metrics

    def _evaluate_regression(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate regression metrics."""
        metrics = {}

        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)

        # MAPE (avoid division by zero)
        try:
            metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred) * 100
        except:
            metrics['mape'] = 0.0

        # Additional metrics
        errors = y_true - y_pred
        metrics['mean_error'] = np.mean(errors)
        metrics['std_error'] = np.std(errors)

        # Directional accuracy (for financial predictions)
        if len(y_true) > 1:
            true_direction = np.sign(np.diff(y_true))
            pred_direction = np.sign(np.diff(y_pred))
            metrics['directional_accuracy'] = accuracy_score(
                true_direction, pred_direction
            )

        return metrics

    def compare_models(
        self,
        results: Dict[str, EvaluationResult],
        primary_metric: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Compare multiple model results.

        Args:
            results: Dictionary mapping model names to EvaluationResult
            primary_metric: Primary metric for ranking

        Returns:
            DataFrame with model comparison
        """
        if not results:
            return pd.DataFrame()

        # Extract metrics
        comparison_data = []

        for model_name, result in results.items():
            row = {'model': model_name}
            row.update(result.metrics)
            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)

        # Sort by primary metric
        if primary_metric and primary_metric in df.columns:
            df = df.sort_values(primary_metric, ascending=False)
        elif self.task == 'classification' and 'accuracy' in df.columns:
            df = df.sort_values('accuracy', ascending=False)
        elif self.task == 'regression' and 'r2' in df.columns:
            df = df.sort_values('r2', ascending=False)

        self.logger.info(f"Model comparison:\n{df}")

        return df

    def calculate_trading_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        returns: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate trading-specific metrics.

        Args:
            y_true: True labels (1=up, 0=down)
            y_pred: Predicted labels
            returns: Optional returns data

        Returns:
            Dictionary of trading metrics
        """
        metrics = {}

        # Directional accuracy
        metrics['directional_accuracy'] = accuracy_score(y_true, y_pred)

        # If returns provided, calculate strategy returns
        if returns is not None:
            # Strategy returns based on predictions
            strategy_returns = returns * (2 * y_pred - 1)  # Convert 0/1 to -1/1

            metrics['total_return'] = np.sum(strategy_returns)
            metrics['mean_return'] = np.mean(strategy_returns)
            metrics['std_return'] = np.std(strategy_returns)

            # Sharpe ratio (annualized)
            if metrics['std_return'] > 0:
                metrics['sharpe_ratio'] = (
                    metrics['mean_return'] / metrics['std_return'] * np.sqrt(252)
                )
            else:
                metrics['sharpe_ratio'] = 0.0

            # Win rate
            winning_trades = strategy_returns > 0
            metrics['win_rate'] = np.mean(winning_trades) * 100

            # Max drawdown
            cumulative = np.cumsum(strategy_returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = cumulative - running_max
            metrics['max_drawdown'] = np.min(drawdown)

        return metrics

    def cross_validate_evaluate(
        self,
        cv_predictions: List[Tuple[np.ndarray, np.ndarray]],
        y_pred_proba_list: Optional[List[np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate cross-validation results.

        Args:
            cv_predictions: List of (y_true, y_pred) tuples for each fold
            y_pred_proba_list: Optional list of probability predictions

        Returns:
            Dictionary with aggregated CV metrics
        """
        fold_metrics = []

        for i, (y_true, y_pred) in enumerate(cv_predictions):
            y_proba = y_pred_proba_list[i] if y_pred_proba_list else None

            result = self.evaluate(y_true, y_pred, y_proba)
            fold_metrics.append(result.metrics)

        # Aggregate metrics
        aggregated = {}

        if fold_metrics:
            all_metric_names = fold_metrics[0].keys()

            for metric_name in all_metric_names:
                values = [fold[metric_name] for fold in fold_metrics]
                aggregated[f'{metric_name}_mean'] = np.mean(values)
                aggregated[f'{metric_name}_std'] = np.std(values)

        self.logger.info(
            f"CV Evaluation ({len(fold_metrics)} folds):\n"
            f"  Accuracy: {aggregated.get('accuracy_mean', 0):.4f} "
            f"(+/- {aggregated.get('accuracy_std', 0):.4f})"
        )

        return {
            'fold_metrics': fold_metrics,
            'aggregated_metrics': aggregated,
            'n_folds': len(fold_metrics)
        }
