"""
Prometheus metrics for QuantCLI monitoring.

Exports metrics for:
- Model drift detection
- Model performance
- Feature distributions
- Retraining events
- System health
"""

from prometheus_client import (
    Counter, Gauge, Histogram, Info,
    CollectorRegistry, push_to_gateway,
    start_http_server
)
from typing import Dict, Optional, List
from datetime import datetime
from pathlib import Path

from src.core.logging_config import get_logger

logger = get_logger(__name__)


class QuantMetrics:
    """
    Centralized Prometheus metrics for QuantCLI.

    Usage:
        >>> metrics = QuantMetrics()
        >>> metrics.record_drift(psi=0.15, drifted_features=3)
        >>> metrics.record_prediction(latency_ms=50)
        >>> metrics.start_http_server(port=8000)
    """

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """
        Initialize metrics with optional custom registry.

        Args:
            registry: Custom CollectorRegistry (None = default)
        """
        self.registry = registry
        self.logger = logger

        # ==================== Drift Detection Metrics ====================

        self.drift_psi = Gauge(
            'quantcli_drift_psi',
            'Overall Population Stability Index (PSI) for drift detection',
            registry=self.registry
        )

        self.drift_feature_psi = Gauge(
            'quantcli_drift_feature_psi',
            'Per-feature PSI values',
            ['feature_name'],
            registry=self.registry
        )

        self.drift_drifted_features = Gauge(
            'quantcli_drift_drifted_features_count',
            'Number of features showing significant drift',
            registry=self.registry
        )

        self.drift_checks_total = Counter(
            'quantcli_drift_checks_total',
            'Total number of drift detection checks performed',
            ['status'],  # 'ok', 'drift_detected', 'error'
            registry=self.registry
        )

        self.drift_last_check = Gauge(
            'quantcli_drift_last_check_timestamp',
            'Unix timestamp of last drift detection check',
            registry=self.registry
        )

        # ==================== Retraining Metrics ====================

        self.retraining_triggered = Counter(
            'quantcli_retraining_triggered_total',
            'Total number of model retraining events triggered',
            ['reason'],  # 'drift', 'schedule', 'manual', 'performance'
            registry=self.registry
        )

        self.retraining_duration = Histogram(
            'quantcli_retraining_duration_seconds',
            'Duration of model retraining in seconds',
            buckets=[60, 300, 600, 1800, 3600, 7200],  # 1m to 2h
            registry=self.registry
        )

        self.retraining_last_time = Gauge(
            'quantcli_retraining_last_timestamp',
            'Unix timestamp of last retraining',
            registry=self.registry
        )

        self.retraining_status = Gauge(
            'quantcli_retraining_status',
            'Current retraining status (0=idle, 1=running, 2=failed)',
            registry=self.registry
        )

        # ==================== Model Performance Metrics ====================

        self.model_performance = Gauge(
            'quantcli_model_performance',
            'Model performance metrics',
            ['model_name', 'metric_type'],  # metric_type: accuracy, precision, sharpe, etc.
            registry=self.registry
        )

        self.predictions_total = Counter(
            'quantcli_predictions_total',
            'Total number of predictions made',
            ['model_name', 'status'],  # status: success, error
            registry=self.registry
        )

        self.prediction_latency = Histogram(
            'quantcli_prediction_latency_ms',
            'Prediction latency in milliseconds',
            ['model_name'],
            buckets=[10, 50, 100, 250, 500, 1000, 2000],
            registry=self.registry
        )

        self.model_version = Info(
            'quantcli_model_version',
            'Current model version information',
            registry=self.registry
        )

        # ==================== Feature Engineering Metrics ====================

        self.feature_generation_duration = Histogram(
            'quantcli_feature_generation_duration_seconds',
            'Time to generate features',
            buckets=[1, 5, 10, 30, 60, 120],
            registry=self.registry
        )

        self.feature_count = Gauge(
            'quantcli_feature_count',
            'Number of features generated',
            registry=self.registry
        )

        self.feature_null_count = Gauge(
            'quantcli_feature_null_count',
            'Number of null values in features',
            ['feature_name'],
            registry=self.registry
        )

        # ==================== Data Quality Metrics ====================

        self.data_freshness = Gauge(
            'quantcli_data_freshness_seconds',
            'Age of most recent data in seconds',
            ['data_source'],
            registry=self.registry
        )

        self.data_fetch_errors = Counter(
            'quantcli_data_fetch_errors_total',
            'Total number of data fetch errors',
            ['provider', 'error_type'],
            registry=self.registry
        )

        # ==================== System Health Metrics ====================

        self.system_health = Gauge(
            'quantcli_system_health',
            'Overall system health score (0-100)',
            registry=self.registry
        )

        self.component_status = Gauge(
            'quantcli_component_status',
            'Component status (1=healthy, 0=unhealthy)',
            ['component'],  # database, redis, kafka, ibkr, etc.
            registry=self.registry
        )

        self.logger.info("✓ Prometheus metrics initialized")

    # ==================== Drift Recording Methods ====================

    def record_drift_check(
        self,
        psi: float,
        feature_psi: Dict[str, float],
        drifted_features: List[str],
        is_significant: bool
    ) -> None:
        """
        Record drift detection check results.

        Args:
            psi: Overall PSI value
            feature_psi: Per-feature PSI values
            drifted_features: List of drifted feature names
            is_significant: Whether drift is significant
        """
        try:
            # Record overall PSI
            self.drift_psi.set(psi)

            # Record per-feature PSI
            for feature, psi_value in feature_psi.items():
                self.drift_feature_psi.labels(feature_name=feature).set(psi_value)

            # Record drifted feature count
            self.drift_drifted_features.set(len(drifted_features))

            # Record check count
            status = 'drift_detected' if is_significant else 'ok'
            self.drift_checks_total.labels(status=status).inc()

            # Update last check timestamp
            self.drift_last_check.set(datetime.now().timestamp())

            self.logger.info(
                f"Drift metrics recorded: PSI={psi:.4f}, "
                f"drifted={len(drifted_features)}, significant={is_significant}"
            )
        except Exception as e:
            self.logger.error(f"Failed to record drift metrics: {e}")
            self.drift_checks_total.labels(status='error').inc()

    def record_drift_error(self, error_type: str) -> None:
        """Record drift detection error."""
        self.drift_checks_total.labels(status='error').inc()
        self.logger.warning(f"Drift detection error: {error_type}")

    # ==================== Retraining Recording Methods ====================

    def record_retraining_triggered(self, reason: str) -> None:
        """
        Record that retraining was triggered.

        Args:
            reason: Reason for retraining (drift, schedule, manual, performance)
        """
        self.retraining_triggered.labels(reason=reason).inc()
        self.retraining_status.set(1)  # Running
        self.logger.info(f"Retraining triggered: reason={reason}")

    def record_retraining_completed(self, duration_seconds: float, success: bool = True) -> None:
        """
        Record retraining completion.

        Args:
            duration_seconds: Duration of retraining
            success: Whether retraining succeeded
        """
        self.retraining_duration.observe(duration_seconds)
        self.retraining_last_time.set(datetime.now().timestamp())
        self.retraining_status.set(0 if success else 2)  # Idle or Failed

        status = "succeeded" if success else "failed"
        self.logger.info(f"Retraining {status} in {duration_seconds:.1f}s")

    # ==================== Model Performance Methods ====================

    def record_model_performance(
        self,
        model_name: str,
        metrics: Dict[str, float]
    ) -> None:
        """
        Record model performance metrics.

        Args:
            model_name: Name of the model
            metrics: Dictionary of metric_name -> value
        """
        for metric_type, value in metrics.items():
            self.model_performance.labels(
                model_name=model_name,
                metric_type=metric_type
            ).set(value)

        self.logger.info(f"Model performance recorded for {model_name}")

    def record_prediction(
        self,
        model_name: str,
        latency_ms: float,
        success: bool = True
    ) -> None:
        """
        Record a prediction event.

        Args:
            model_name: Name of the model
            latency_ms: Prediction latency in milliseconds
            success: Whether prediction succeeded
        """
        status = 'success' if success else 'error'
        self.predictions_total.labels(model_name=model_name, status=status).inc()

        if success:
            self.prediction_latency.labels(model_name=model_name).observe(latency_ms)

    def set_model_version(self, model_name: str, version: str, trained_at: str) -> None:
        """
        Set current model version information.

        Args:
            model_name: Name of the model
            version: Model version
            trained_at: Timestamp when model was trained
        """
        self.model_version.info({
            'model_name': model_name,
            'version': version,
            'trained_at': trained_at
        })

    # ==================== Feature Engineering Methods ====================

    def record_feature_generation(self, duration_seconds: float, feature_count: int) -> None:
        """Record feature generation metrics."""
        self.feature_generation_duration.observe(duration_seconds)
        self.feature_count.set(feature_count)

    def record_feature_nulls(self, feature_name: str, null_count: int) -> None:
        """Record null counts per feature."""
        self.feature_null_count.labels(feature_name=feature_name).set(null_count)

    # ==================== Data Quality Methods ====================

    def record_data_freshness(self, data_source: str, age_seconds: float) -> None:
        """Record data freshness."""
        self.data_freshness.labels(data_source=data_source).set(age_seconds)

    def record_data_fetch_error(self, provider: str, error_type: str) -> None:
        """Record data fetch error."""
        self.data_fetch_errors.labels(provider=provider, error_type=error_type).inc()

    # ==================== System Health Methods ====================

    def set_system_health(self, health_score: float) -> None:
        """Set overall system health (0-100)."""
        self.system_health.set(health_score)

    def set_component_status(self, component: str, is_healthy: bool) -> None:
        """Set component health status."""
        self.component_status.labels(component=component).set(1 if is_healthy else 0)

    # ==================== Server Management ====================

    def start_http_server(self, port: int = 8000) -> None:
        """
        Start Prometheus metrics HTTP server.

        Args:
            port: Port to listen on (default: 8000)
        """
        try:
            start_http_server(port, registry=self.registry)
            self.logger.success(f"✓ Prometheus metrics server started on port {port}")
        except OSError as e:
            if "Address already in use" in str(e):
                self.logger.warning(f"Port {port} already in use (server may already be running)")
            else:
                raise

    def push_to_gateway(
        self,
        gateway_url: str,
        job: str = 'quantcli'
    ) -> None:
        """
        Push metrics to Prometheus Pushgateway.

        Args:
            gateway_url: URL of Pushgateway (e.g., 'localhost:9091')
            job: Job label for metrics
        """
        try:
            push_to_gateway(gateway_url, job=job, registry=self.registry)
            self.logger.info(f"Metrics pushed to {gateway_url}")
        except Exception as e:
            self.logger.error(f"Failed to push metrics: {e}")


# Global metrics instance (singleton pattern)
_metrics_instance: Optional[QuantMetrics] = None


def get_metrics() -> QuantMetrics:
    """
    Get global metrics instance (singleton).

    Returns:
        Global QuantMetrics instance
    """
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = QuantMetrics()
    return _metrics_instance


def init_metrics(port: Optional[int] = None) -> QuantMetrics:
    """
    Initialize global metrics and optionally start HTTP server.

    Args:
        port: Port for HTTP server (None = don't start server)

    Returns:
        Initialized QuantMetrics instance
    """
    metrics = get_metrics()

    if port is not None:
        metrics.start_http_server(port)

    return metrics
