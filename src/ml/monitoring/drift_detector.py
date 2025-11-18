"""
Model Monitoring and Drift Detection

Monitors models in production for:
- Data drift (feature distribution changes)
- Concept drift (target relationship changes)
- Performance degradation
- Triggers for retraining
"""

from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats


class DriftDetector:
    """
    Detect data and concept drift.

    Methods:
    - PSI (Population Stability Index): Feature distribution changes
    - KS (Kolmogorov-Smirnov): Statistical difference between distributions
    - JSD (Jensen-Shannon Divergence): Symmetric KL divergence
    """

    def __init__(
        self,
        psi_threshold: float = 0.1,
        ks_threshold: float = 0.05,
        jsd_threshold: float = 0.05,
    ):
        """
        Initialize drift detector.

        Args:
            psi_threshold: PSI threshold (0.1 = moderate drift, 0.25 = significant)
            ks_threshold: KS test p-value threshold
            jsd_threshold: JSD threshold
        """
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold
        self.jsd_threshold = jsd_threshold

    def calculate_psi(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
        bins: int = 10,
    ) -> float:
        """
        Calculate Population Stability Index (PSI).

        PSI < 0.1: No drift
        0.1 <= PSI < 0.25: Moderate drift
        PSI >= 0.25: Significant drift

        Args:
            baseline: Baseline distribution
            current: Current distribution
            bins: Number of bins for discretization

        Returns:
            PSI value
        """
        # Create bins based on baseline
        breakpoints = np.quantile(baseline, np.linspace(0, 1, bins + 1))
        breakpoints = np.unique(breakpoints)  # Remove duplicates

        # Calculate frequencies
        baseline_freq, _ = np.histogram(baseline, bins=breakpoints)
        current_freq, _ = np.histogram(current, bins=breakpoints)

        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        baseline_pct = (baseline_freq + epsilon) / (len(baseline) + epsilon * bins)
        current_pct = (current_freq + epsilon) / (len(current) + epsilon * bins)

        # PSI formula
        psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))

        return float(psi)

    def calculate_ks_statistic(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Calculate Kolmogorov-Smirnov statistic.

        Args:
            baseline: Baseline distribution
            current: Current distribution

        Returns:
            (ks_statistic, p_value)
        """
        ks_stat, p_value = stats.ks_2samp(baseline, current)
        return float(ks_stat), float(p_value)

    def calculate_jsd(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
        bins: int = 50,
    ) -> float:
        """
        Calculate Jensen-Shannon Divergence.

        JSD is a symmetric version of KL divergence.
        Range: [0, 1] where 0 = identical, 1 = completely different

        Args:
            baseline: Baseline distribution
            current: Current distribution
            bins: Number of bins

        Returns:
            JSD value
        """
        # Create histograms
        min_val = min(baseline.min(), current.min())
        max_val = max(baseline.max(), current.max())
        bins_edges = np.linspace(min_val, max_val, bins + 1)

        p, _ = np.histogram(baseline, bins=bins_edges, density=True)
        q, _ = np.histogram(current, bins=bins_edges, density=True)

        # Normalize
        p = p / (p.sum() + 1e-10)
        q = q / (q.sum() + 1e-10)

        # Calculate JSD
        m = 0.5 * (p + q)

        # KL divergence
        def kl_div(p, q):
            return np.sum(np.where(p != 0, p * np.log((p + 1e-10) / (q + 1e-10)), 0))

        jsd = 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)

        return float(jsd)

    def detect_feature_drift(
        self,
        baseline_df: pd.DataFrame,
        current_df: pd.DataFrame,
        features: List[str] = None,
    ) -> Dict:
        """
        Detect drift for all features.

        Args:
            baseline_df: Baseline feature data
            current_df: Current feature data
            features: List of features to check (None = all)

        Returns:
            Dict with drift metrics per feature
        """
        if features is None:
            features = baseline_df.columns.tolist()

        results = {
            "timestamp": datetime.now().isoformat(),
            "n_features_checked": len(features),
            "features": {},
            "drift_detected": False,
            "drifted_features": [],
        }

        for feature in features:
            if feature not in baseline_df.columns or feature not in current_df.columns:
                logger.warning(f"Feature {feature} not found in data")
                continue

            baseline = baseline_df[feature].dropna().values
            current = current_df[feature].dropna().values

            if len(baseline) < 30 or len(current) < 30:
                logger.warning(f"Insufficient data for {feature}")
                continue

            # Calculate metrics
            psi = self.calculate_psi(baseline, current)
            ks_stat, ks_pval = self.calculate_ks_statistic(baseline, current)
            jsd = self.calculate_jsd(baseline, current)

            # Detect drift
            drift_detected = (
                psi > self.psi_threshold
                or ks_pval < self.ks_threshold
                or jsd > self.jsd_threshold
            )

            feature_result = {
                "psi": psi,
                "ks_statistic": ks_stat,
                "ks_pvalue": ks_pval,
                "jsd": jsd,
                "drift_detected": drift_detected,
                "baseline_mean": float(baseline.mean()),
                "current_mean": float(current.mean()),
                "baseline_std": float(baseline.std()),
                "current_std": float(current.std()),
            }

            results["features"][feature] = feature_result

            if drift_detected:
                results["drifted_features"].append(feature)
                logger.warning(
                    f"Drift detected in {feature}: PSI={psi:.4f}, KS_p={ks_pval:.4f}, JSD={jsd:.4f}"
                )

        results["drift_detected"] = len(results["drifted_features"]) > 0

        # Summary
        if results["drift_detected"]:
            logger.warning(
                f"⚠️  Drift detected in {len(results['drifted_features'])} features: "
                f"{results['drifted_features']}"
            )
        else:
            logger.success("✅ No significant drift detected")

        return results


class PerformanceMonitor:
    """
    Monitor model performance in production.

    Tracks:
    - Prediction accuracy
    - Sharpe ratio (rolling)
    - Drawdown
    - Win rate
    - Latency
    """

    def __init__(
        self,
        window_size: int = 1000,
        min_sharpe: float = 1.0,
        max_drawdown: float = -0.25,
    ):
        """
        Initialize performance monitor.

        Args:
            window_size: Rolling window for metrics
            min_sharpe: Minimum acceptable Sharpe ratio
            max_drawdown: Maximum acceptable drawdown
        """
        self.window_size = window_size
        self.min_sharpe = min_sharpe
        self.max_drawdown = max_drawdown

        # Storage
        self.predictions = []
        self.actuals = []
        self.timestamps = []

    def log_prediction(
        self,
        timestamp: datetime,
        prediction: float,
        actual: float = None,
    ):
        """
        Log a prediction (and actual if available).

        Args:
            timestamp: Prediction timestamp
            prediction: Model prediction
            actual: Actual outcome (if available)
        """
        self.timestamps.append(timestamp)
        self.predictions.append(prediction)
        self.actuals.append(actual)

        # Keep only recent window
        if len(self.predictions) > self.window_size * 2:
            self.timestamps = self.timestamps[-self.window_size:]
            self.predictions = self.predictions[-self.window_size:]
            self.actuals = self.actuals[-self.window_size:]

    def calculate_metrics(self) -> Dict:
        """
        Calculate performance metrics on recent predictions.

        Returns:
            Dict with metrics
        """
        if len(self.predictions) < 100:
            logger.warning("Insufficient predictions for metrics")
            return {}

        # Filter where actuals are available
        valid_idx = [i for i, a in enumerate(self.actuals) if a is not None]
        if len(valid_idx) < 100:
            logger.warning("Insufficient actuals for metrics")
            return {}

        preds = np.array([self.predictions[i] for i in valid_idx])
        actuals = np.array([self.actuals[i] for i in valid_idx])

        # Assuming predictions and actuals are returns
        returns = actuals  # Actual returns

        # Sharpe Ratio
        sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)

        # Max Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()

        # Win Rate
        win_rate = (returns > 0).sum() / len(returns)

        # Prediction accuracy (direction)
        direction_accuracy = ((preds > 0) == (actuals > 0)).sum() / len(preds)

        # Correlation
        correlation = np.corrcoef(preds, actuals)[0, 1]

        metrics = {
            "timestamp": datetime.now().isoformat(),
            "n_predictions": len(preds),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_dd),
            "win_rate": float(win_rate),
            "direction_accuracy": float(direction_accuracy),
            "correlation": float(correlation),
            "mean_prediction": float(preds.mean()),
            "mean_actual": float(actuals.mean()),
        }

        # Check thresholds
        metrics["performance_degraded"] = (
            sharpe < self.min_sharpe or max_dd < self.max_drawdown
        )

        if metrics["performance_degraded"]:
            logger.warning(
                f"⚠️  Performance degraded: Sharpe={sharpe:.2f}, Drawdown={max_dd:.2%}"
            )
        else:
            logger.success(f"✅ Performance healthy: Sharpe={sharpe:.2f}")

        return metrics


class RetrainingTrigger:
    """
    Determine when to retrain models.

    Triggers:
    - Scheduled (quarterly, monthly, weekly)
    - Drift detected
    - Performance degraded
    - Minimum samples accumulated
    """

    def __init__(
        self,
        schedule_days: int = 90,  # Quarterly
        min_samples: int = 10000,
        drift_threshold: int = 3,  # Number of drifted features
    ):
        """
        Initialize retraining trigger.

        Args:
            schedule_days: Days between scheduled retraining
            min_samples: Minimum new samples before retraining
            drift_threshold: Number of drifted features to trigger retraining
        """
        self.schedule_days = schedule_days
        self.min_samples = min_samples
        self.drift_threshold = drift_threshold

        self.last_training_date = None
        self.samples_since_training = 0

    def should_retrain(
        self,
        drift_results: Dict = None,
        performance_metrics: Dict = None,
    ) -> Tuple[bool, List[str]]:
        """
        Determine if retraining should be triggered.

        Args:
            drift_results: Results from DriftDetector
            performance_metrics: Results from PerformanceMonitor

        Returns:
            (should_retrain, reasons)
        """
        reasons = []
        should_retrain = False

        # Check scheduled retraining
        if self.last_training_date:
            days_since = (datetime.now() - self.last_training_date).days
            if days_since >= self.schedule_days:
                reasons.append(f"Scheduled retraining ({days_since} days)")
                should_retrain = True

        # Check drift
        if drift_results and drift_results.get("drift_detected"):
            n_drifted = len(drift_results.get("drifted_features", []))
            if n_drifted >= self.drift_threshold:
                reasons.append(f"Drift detected ({n_drifted} features)")
                should_retrain = True

        # Check performance
        if performance_metrics and performance_metrics.get("performance_degraded"):
            reasons.append("Performance degraded")
            should_retrain = True

        # Check sample accumulation
        if self.samples_since_training >= self.min_samples:
            reasons.append(f"New samples accumulated ({self.samples_since_training})")
            should_retrain = True

        if should_retrain:
            logger.warning(f"🔄 Retraining triggered: {', '.join(reasons)}")
        else:
            logger.info("No retraining needed")

        return should_retrain, reasons

    def mark_training_completed(self):
        """Mark that retraining was completed."""
        self.last_training_date = datetime.now()
        self.samples_since_training = 0
        logger.success(f"Retraining completed at {self.last_training_date}")


# Example monitoring workflow
def monitoring_workflow_example():
    """Example of production monitoring workflow."""

    # Initialize monitors
    drift_detector = DriftDetector(
        psi_threshold=0.1,
        ks_threshold=0.05,
        jsd_threshold=0.05,
    )

    performance_monitor = PerformanceMonitor(
        window_size=1000,
        min_sharpe=1.0,
        max_drawdown=-0.25,
    )

    retraining_trigger = RetrainingTrigger(
        schedule_days=90,
        min_samples=10000,
        drift_threshold=3,
    )

    # Simulate daily monitoring
    # for day in range(30):
    #     # Get new data
    #     current_features = get_today_features()
    #
    #     # Check drift (compare to baseline)
    #     drift_results = drift_detector.detect_feature_drift(
    #         baseline_df=baseline_features,
    #         current_df=current_features,
    #     )
    #
    #     # Log predictions
    #     for symbol, pred, actual in daily_predictions:
    #         performance_monitor.log_prediction(
    #             timestamp=datetime.now(),
    #             prediction=pred,
    #             actual=actual,
    #         )
    #
    #     # Calculate performance
    #     perf_metrics = performance_monitor.calculate_metrics()
    #
    #     # Check if retraining needed
    #     should_retrain, reasons = retraining_trigger.should_retrain(
    #         drift_results=drift_results,
    #         performance_metrics=perf_metrics,
    #     )
    #
    #     if should_retrain:
    #         trigger_retraining_pipeline()
    #         retraining_trigger.mark_training_completed()

    logger.info("Monitoring workflow example (commented out)")
