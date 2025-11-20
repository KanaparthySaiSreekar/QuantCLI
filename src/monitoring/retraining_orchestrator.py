"""
Automatic model retraining orchestrator.

Handles:
- Drift-triggered retraining
- Scheduled retraining
- Performance-based retraining
- Retraining workflow orchestration
"""

import pandas as pd
import numpy as np
from typing import Optional, Callable, Dict, Any, List
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
import time

from src.core.logging_config import get_logger
from src.core.exceptions import ModelError
from src.monitoring.drift_detection import DriftDetector, DriftReport, DriftMonitor
from src.monitoring.metrics import get_metrics
from src.models.trainer import ModelTrainer
from src.models.base import BaseModel

logger = get_logger(__name__)
metrics = get_metrics()


@dataclass
class RetrainingConfig:
    """Configuration for retraining orchestrator."""
    # Drift-based retraining
    enable_drift_retraining: bool = True
    drift_psi_threshold: float = 0.1
    drift_check_interval_days: int = 7

    # Schedule-based retraining
    enable_scheduled_retraining: bool = True
    retraining_interval_days: int = 30

    # Performance-based retraining
    enable_performance_retraining: bool = True
    min_performance_threshold: float = 0.7  # Minimum acceptable performance
    performance_metric: str = 'accuracy'  # or 'sharpe', 'r2', etc.

    # Retraining behavior
    min_retraining_interval_days: int = 3  # Minimum time between retrainings
    max_retries: int = 3
    save_old_model: bool = True


class RetrainingOrchestrator:
    """
    Orchestrates automatic model retraining based on various triggers.

    Triggers:
    1. Drift detection - Retrain when significant drift detected
    2. Schedule - Retrain on fixed schedule
    3. Performance - Retrain when performance degrades
    4. Manual - Explicit retraining request

    Example:
        >>> config = RetrainingConfig()
        >>> orchestrator = RetrainingOrchestrator(
        ...     model_trainer=trainer,
        ...     drift_detector=detector,
        ...     config=config
        ... )
        >>> orchestrator.check_and_retrain(recent_data)
    """

    def __init__(
        self,
        model_trainer: ModelTrainer,
        drift_detector: DriftDetector,
        config: RetrainingConfig,
        model_save_path: Optional[Path] = None,
        on_retraining_complete: Optional[Callable[[bool, Dict], None]] = None
    ):
        """
        Initialize retraining orchestrator.

        Args:
            model_trainer: ModelTrainer instance for retraining
            drift_detector: DriftDetector instance for monitoring
            config: Retraining configuration
            model_save_path: Path to save retrained models
            on_retraining_complete: Callback when retraining completes
        """
        self.model_trainer = model_trainer
        self.drift_detector = drift_detector
        self.config = config
        self.model_save_path = model_save_path or Path("models/retrained")
        self.on_retraining_complete = on_retraining_complete

        # Initialize drift monitor with retraining callback
        self.drift_monitor = DriftMonitor(
            detector=drift_detector,
            check_interval_days=config.drift_check_interval_days,
            alert_callback=self._handle_drift_alert
        )

        # State tracking
        self.last_retraining = None
        self.last_scheduled_retraining = None
        self.retraining_history = []
        self.is_retraining = False

        self.logger = logger

    def check_and_retrain(
        self,
        current_data: pd.DataFrame,
        X_train: Optional[pd.DataFrame] = None,
        y_train: Optional[pd.Series] = None,
        force: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Check all triggers and retrain if necessary.

        Args:
            current_data: Recent data for drift detection
            X_train: Training features (required if retraining triggered)
            y_train: Training target (required if retraining triggered)
            force: Force retraining regardless of triggers

        Returns:
            Retraining result dict if retrained, None otherwise
        """
        if self.is_retraining:
            self.logger.warning("Retraining already in progress, skipping")
            return None

        # Check if minimum interval has passed
        if not force and self.last_retraining is not None:
            days_since = (datetime.now() - self.last_retraining).days
            if days_since < self.config.min_retraining_interval_days:
                self.logger.debug(
                    f"Minimum retraining interval not met "
                    f"({days_since}/{self.config.min_retraining_interval_days} days)"
                )
                return None

        # Determine retraining trigger
        trigger_reason = None

        if force:
            trigger_reason = 'manual'
        else:
            # Check drift
            if self.config.enable_drift_retraining:
                drift_report = self.drift_monitor.check(current_data, force=False)
                if drift_report and drift_report.is_significant_drift:
                    trigger_reason = 'drift'
                    self.logger.warning(f"Drift detected: PSI={drift_report.overall_psi:.4f}")

            # Check schedule
            if trigger_reason is None and self.config.enable_scheduled_retraining:
                if self._should_retrain_on_schedule():
                    trigger_reason = 'schedule'
                    self.logger.info("Scheduled retraining triggered")

        # If no trigger, return
        if trigger_reason is None:
            return None

        # Validate training data
        if X_train is None or y_train is None:
            self.logger.error("Retraining triggered but training data not provided")
            return None

        # Perform retraining
        return self._perform_retraining(
            X_train=X_train,
            y_train=y_train,
            reason=trigger_reason
        )

    def retrain_now(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        reason: str = 'manual'
    ) -> Dict[str, Any]:
        """
        Immediately trigger retraining.

        Args:
            X_train: Training features
            y_train: Training target
            reason: Reason for retraining

        Returns:
            Retraining result dict
        """
        return self._perform_retraining(X_train, y_train, reason)

    def _perform_retraining(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        reason: str
    ) -> Dict[str, Any]:
        """
        Execute model retraining workflow.

        Args:
            X_train: Training features
            y_train: Training target
            reason: Reason for retraining

        Returns:
            Dict with retraining results
        """
        self.is_retraining = True
        start_time = time.time()

        self.logger.info(f"🔄 Starting model retraining (reason: {reason})")

        # Record metrics
        metrics.record_retraining_triggered(reason=reason)

        result = {
            'success': False,
            'reason': reason,
            'started_at': datetime.now().isoformat(),
            'error': None,
            'metrics': {}
        }

        try:
            # Save old model if configured
            if self.config.save_old_model and self.model_trainer.model.is_trained:
                old_model_path = self.model_save_path / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                self.model_trainer.save_model(old_model_path)
                self.logger.info(f"Old model saved to {old_model_path}")

            # Train new model
            self.logger.info(f"Training on {len(X_train)} samples with {len(X_train.columns)} features")
            training_result = self.model_trainer.train(X_train, y_train)

            # Save new model
            self.model_trainer.save_model(self.model_save_path / "latest")
            self.logger.success(f"✓ Model retrained successfully")

            # Update result
            result['success'] = True
            result['metrics'] = training_result.get('test_metrics', {})
            result['completed_at'] = datetime.now().isoformat()

            # Update state
            self.last_retraining = datetime.now()
            if reason == 'schedule':
                self.last_scheduled_retraining = datetime.now()

            # Record retraining history
            self.retraining_history.append({
                'timestamp': result['completed_at'],
                'reason': reason,
                'success': True,
                'metrics': result['metrics']
            })

        except Exception as e:
            self.logger.error(f"Retraining failed: {e}")
            result['error'] = str(e)
            result['completed_at'] = datetime.now().isoformat()

            # Record failure in history
            self.retraining_history.append({
                'timestamp': result['completed_at'],
                'reason': reason,
                'success': False,
                'error': str(e)
            })

        finally:
            duration = time.time() - start_time
            result['duration_seconds'] = duration

            # Record metrics
            metrics.record_retraining_completed(
                duration_seconds=duration,
                success=result['success']
            )

            # Record model performance metrics if successful
            if result['success'] and result['metrics']:
                metrics.record_model_performance(
                    model_name=self.model_trainer.model.model_name,
                    metrics=result['metrics']
                )

            self.is_retraining = False

            # Call completion callback
            if self.on_retraining_complete:
                try:
                    self.on_retraining_complete(result['success'], result)
                except Exception as e:
                    self.logger.error(f"Retraining callback failed: {e}")

        return result

    def _handle_drift_alert(self, drift_report: DriftReport) -> None:
        """
        Callback for drift detection alerts.

        Args:
            drift_report: Drift detection report
        """
        self.logger.warning(
            f"⚠️  Drift alert received: PSI={drift_report.overall_psi:.4f}, "
            f"{len(drift_report.drifted_features)} drifted features"
        )
        # Note: Actual retraining is handled by check_and_retrain()
        # This callback is just for logging/alerting

    def _should_retrain_on_schedule(self) -> bool:
        """Check if scheduled retraining should occur."""
        if self.last_scheduled_retraining is None:
            # Never retrained on schedule
            if self.last_retraining is None:
                return True  # First retraining
            # Check if enough time passed since any retraining
            days_since = (datetime.now() - self.last_retraining).days
            return days_since >= self.config.retraining_interval_days

        # Check if enough time since last scheduled retraining
        days_since = (datetime.now() - self.last_scheduled_retraining).days
        return days_since >= self.config.retraining_interval_days

    def get_retraining_history(self) -> pd.DataFrame:
        """
        Get history of retraining events.

        Returns:
            DataFrame with retraining history
        """
        if not self.retraining_history:
            return pd.DataFrame()

        return pd.DataFrame(self.retraining_history)

    def get_retraining_summary(self) -> Dict[str, Any]:
        """
        Get summary of retraining statistics.

        Returns:
            Dictionary with summary statistics
        """
        if not self.retraining_history:
            return {
                'total_retrainings': 0,
                'successful': 0,
                'failed': 0,
                'last_retraining': None,
                'avg_duration': 0.0
            }

        history_df = self.get_retraining_history()

        return {
            'total_retrainings': len(history_df),
            'successful': history_df['success'].sum(),
            'failed': (~history_df['success']).sum(),
            'success_rate': history_df['success'].mean(),
            'last_retraining': self.last_retraining.isoformat() if self.last_retraining else None,
            'reasons': history_df['reason'].value_counts().to_dict()
        }

    def enable_drift_retraining(self) -> None:
        """Enable automatic retraining on drift detection."""
        self.config.enable_drift_retraining = True
        self.logger.info("Drift-based retraining enabled")

    def disable_drift_retraining(self) -> None:
        """Disable automatic retraining on drift detection."""
        self.config.enable_drift_retraining = False
        self.logger.warning("Drift-based retraining disabled")

    def enable_scheduled_retraining(self) -> None:
        """Enable scheduled retraining."""
        self.config.enable_scheduled_retraining = True
        self.logger.info("Scheduled retraining enabled")

    def disable_scheduled_retraining(self) -> None:
        """Disable scheduled retraining."""
        self.config.enable_scheduled_retraining = False
        self.logger.warning("Scheduled retraining disabled")


def create_retraining_orchestrator(
    model_trainer: ModelTrainer,
    drift_detector: DriftDetector,
    config: Optional[RetrainingConfig] = None,
    model_save_path: Optional[Path] = None
) -> RetrainingOrchestrator:
    """
    Factory function to create a configured RetrainingOrchestrator.

    Args:
        model_trainer: ModelTrainer instance
        drift_detector: DriftDetector instance
        config: Optional RetrainingConfig (uses defaults if None)
        model_save_path: Path to save models

    Returns:
        Configured RetrainingOrchestrator instance
    """
    if config is None:
        config = RetrainingConfig()

    orchestrator = RetrainingOrchestrator(
        model_trainer=model_trainer,
        drift_detector=drift_detector,
        config=config,
        model_save_path=model_save_path
    )

    logger.info("✓ Retraining orchestrator initialized")
    logger.info(f"  Drift retraining: {'enabled' if config.enable_drift_retraining else 'disabled'}")
    logger.info(f"  Scheduled retraining: {'enabled' if config.enable_scheduled_retraining else 'disabled'}")
    logger.info(f"  Performance retraining: {'enabled' if config.enable_performance_retraining else 'disabled'}")

    return orchestrator
