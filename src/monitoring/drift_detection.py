"""
Population Stability Index (PSI) drift detection.

Monitors feature distributions for significant changes that indicate model degradation.
PSI measures distribution shifts between baseline and current data.

Thresholds:
    - PSI < 0.1: No significant change
    - 0.1 <= PSI < 0.2: Moderate change (monitor)
    - PSI >= 0.2: Significant change (retrain)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.core.logging_config import get_logger
from src.core.exceptions import ValidationError
from src.monitoring.metrics import get_metrics

logger = get_logger(__name__)
metrics = get_metrics()


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

    def __str__(self) -> str:
        """String representation of drift report."""
        status = "⚠️  DRIFT DETECTED" if self.is_significant_drift else "✓ No significant drift"
        return f"""
Drift Detection Report
=====================
Timestamp: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
Status: {status}

Overall PSI: {self.overall_psi:.4f} (threshold: {self.threshold:.2f})
Drifted Features: {len(self.drifted_features)}

Recommendation: {self.recommendation}

Top Drifted Features:
{self._format_top_features()}
"""

    def _format_top_features(self, n: int = 5) -> str:
        """Format top drifted features for display."""
        if not self.feature_psi:
            return "  (none)"

        sorted_features = sorted(
            self.feature_psi.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n]

        lines = []
        for feature, psi in sorted_features:
            status = "⚠️ " if psi >= self.threshold else "  "
            lines.append(f"  {status}{feature}: {psi:.4f}")

        return "\n".join(lines)


class DriftDetector:
    """
    Detects distribution drift using Population Stability Index (PSI).

    PSI Formula:
        PSI = sum((current_pct - baseline_pct) * ln(current_pct / baseline_pct))

    Example:
        >>> detector = DriftDetector(psi_threshold=0.1)
        >>> detector.fit(train_data, feature_cols)
        >>> report = detector.detect(recent_data, feature_cols)
        >>> if report.is_significant_drift:
        ...     trigger_retraining()
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

    def fit(
        self,
        baseline_data: pd.DataFrame,
        feature_cols: Optional[List[str]] = None
    ) -> 'DriftDetector':
        """
        Fit baseline distributions.

        Args:
            baseline_data: Training data to use as baseline
            feature_cols: List of feature columns (None = all numeric)

        Returns:
            Self for chaining
        """
        if feature_cols is None:
            feature_cols = baseline_data.select_dtypes(
                include=[np.number]
            ).columns.tolist()

        self.logger.info(f"Fitting baseline distributions for {len(feature_cols)} features")

        fitted_count = 0
        for feature in feature_cols:
            if feature not in baseline_data.columns:
                self.logger.warning(f"Feature {feature} not in baseline data, skipping")
                continue

            values = baseline_data[feature].dropna()

            if len(values) == 0:
                self.logger.warning(f"Feature {feature} has no values, skipping")
                continue

            # Create bins and calculate distribution
            try:
                bins, dist = self._create_distribution(values)
                self.baseline_distributions[feature] = {
                    'bins': bins,
                    'distribution': dist
                }
                fitted_count += 1
            except Exception as e:
                self.logger.warning(f"Failed to create distribution for {feature}: {e}")

        self.logger.success(
            f"✓ Baseline fitted for {fitted_count}/{len(feature_cols)} features"
        )

        return self

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

        Raises:
            ValidationError: If baseline not fitted
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
                self.logger.warning(f"Feature {feature} has no current values, skipping")
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

        # Calculate percentage of drifted features
        pct_drifted = len(drifted_features) / len(feature_cols) if feature_cols else 0

        # Generate recommendation
        if is_significant or pct_drifted > 0.2:
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
        self.logger.info(
            f"Drift detection: PSI={overall_psi:.4f}, "
            f"{len(drifted_features)}/{len(feature_cols)} drifted features"
        )
        if is_significant:
            self.logger.warning(f"⚠️  {recommendation}")

        # Record Prometheus metrics
        try:
            metrics.record_drift_check(
                psi=overall_psi,
                feature_psi=feature_psi,
                drifted_features=drifted_features,
                is_significant=is_significant
            )
        except Exception as e:
            self.logger.warning(f"Failed to record drift metrics: {e}")

        return report

    def _create_distribution(
        self,
        values: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create binned distribution from values.

        Args:
            values: Series of values to bin

        Returns:
            (bin_edges, distribution_percentages)
        """
        # Use quantile-based binning for better handling of outliers
        try:
            # Try quantile binning first
            bins = pd.qcut(
                values,
                q=self.n_bins,
                duplicates='drop',
                retbins=True
            )[1]
        except ValueError:
            # Fallback to equal-width bins if quantiles fail
            bins = np.linspace(
                values.min(),
                values.max(),
                self.n_bins + 1
            )

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
        psi = np.sum(
            (current_dist - baseline_dist) * np.log(current_dist / baseline_dist)
        )

        return psi

    def save_baseline(self, path: Path) -> None:
        """
        Save baseline distributions to file.

        Args:
            path: Path to save baseline
        """
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.baseline_distributions, path)
        self.logger.info(f"Baseline saved to {path}")

    def load_baseline(self, path: Path) -> 'DriftDetector':
        """
        Load baseline distributions from file.

        Args:
            path: Path to load baseline from

        Returns:
            Self for chaining
        """
        import joblib

        self.baseline_distributions = joblib.load(path)
        self.logger.info(
            f"Baseline loaded from {path} "
            f"({len(self.baseline_distributions)} features)"
        )

        return self


class DriftMonitor:
    """
    Continuously monitors for drift and triggers retraining.

    Usage:
        >>> monitor = DriftMonitor(detector, check_interval_days=7)
        >>> report = monitor.check(recent_data)
        >>> if report and report.is_significant_drift:
        ...     trigger_retraining()
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

    def check(
        self,
        current_data: pd.DataFrame,
        force: bool = False
    ) -> Optional[DriftReport]:
        """
        Check for drift if interval has passed.

        Args:
            current_data: Recent data to check
            force: Force check even if interval hasn't passed

        Returns:
            DriftReport if check was performed, None otherwise
        """
        now = datetime.now()

        # Check if enough time has passed
        if not force and self.last_check is not None:
            days_since_check = (now - self.last_check).days
            if days_since_check < self.check_interval_days:
                self.logger.debug(
                    f"Skipping drift check ({days_since_check}/{self.check_interval_days} days)"
                )
                return None

        # Perform drift detection
        self.logger.info("Performing drift detection check...")
        report = self.detector.detect(current_data)

        # Store history
        self.drift_history.append({
            'timestamp': report.timestamp,
            'psi': report.overall_psi,
            'is_significant': report.is_significant_drift,
            'drifted_features': len(report.drifted_features)
        })

        # Alert if significant drift
        if report.is_significant_drift:
            self.logger.warning(f"⚠️  Drift detected: {report.recommendation}")
            if self.alert_callback:
                try:
                    self.alert_callback(report)
                except Exception as e:
                    self.logger.error(f"Alert callback failed: {e}")

        self.last_check = now

        return report

    def get_drift_history(self) -> pd.DataFrame:
        """
        Get history of drift checks.

        Returns:
            DataFrame with drift history
        """
        if not self.drift_history:
            return pd.DataFrame()

        return pd.DataFrame(self.drift_history)

    def get_drift_summary(self) -> Dict:
        """
        Get summary statistics of drift history.

        Returns:
            Dictionary with summary statistics
        """
        if not self.drift_history:
            return {
                'total_checks': 0,
                'drift_detected': 0,
                'avg_psi': 0.0,
                'max_psi': 0.0
            }

        history_df = self.get_drift_history()

        return {
            'total_checks': len(history_df),
            'drift_detected': history_df['is_significant'].sum(),
            'drift_rate': history_df['is_significant'].mean(),
            'avg_psi': history_df['psi'].mean(),
            'max_psi': history_df['psi'].max(),
            'last_check': history_df['timestamp'].max()
        }
