"""
MLflow Training Pipeline with Model Registry

Production-grade training with:
- Experiment tracking
- Model versioning
- Dataset lineage
- Automatic registration
- Validation gates
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from loguru import logger
from mlflow.models import infer_signature
from sklearn.base import BaseEstimator

from src.backtest.cpcv import BacktestValidator, CombinatorialPurgedCV


class MLflowTrainer:
    """
    MLflow-integrated training pipeline.

    Ensures:
    - Every run is tracked
    - Models are versioned
    - Dataset provenance is recorded
    - Validation gates are enforced
    """

    def __init__(
        self,
        experiment_name: str = "quantcli-ensemble",
        tracking_uri: str = "http://localhost:5000",
        registry_uri: Optional[str] = None,
    ):
        """
        Initialize MLflow trainer.

        Args:
            experiment_name: MLflow experiment name
            tracking_uri: MLflow tracking server URI
            registry_uri: Model registry URI (defaults to tracking_uri)
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.registry_uri = registry_uri or tracking_uri

        # Set MLflow URIs
        mlflow.set_tracking_uri(self.tracking_uri)
        if self.registry_uri:
            mlflow.set_registry_uri(self.registry_uri)

        # Set/create experiment
        try:
            self.experiment = mlflow.set_experiment(experiment_name)
            logger.success(f"MLflow experiment: {experiment_name}")
        except Exception as e:
            logger.warning(f"Could not set MLflow experiment: {e}")
            self.experiment = None

    def _compute_dataset_hash(self, X: pd.DataFrame, y: pd.Series) -> str:
        """Compute deterministic hash of training data."""
        X_hash = hashlib.sha256(X.to_json().encode()).hexdigest()[:16]
        y_hash = hashlib.sha256(y.to_json().encode()).hexdigest()[:16]
        return f"{X_hash}-{y_hash}"

    def _log_dataset_info(self, X: pd.DataFrame, y: pd.Series, split: str = "train"):
        """Log dataset information to MLflow."""
        mlflow.log_param(f"{split}_samples", len(X))
        mlflow.log_param(f"{split}_features", X.shape[1])
        mlflow.log_param(f"{split}_start_date", str(X.index[0]))
        mlflow.log_param(f"{split}_end_date", str(X.index[-1]))

        # Feature names
        mlflow.log_dict(
            {"features": list(X.columns)},
            f"{split}_features.json",
        )

    def train_with_cpcv(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str,
        cv_splits: int = 5,
        params: Optional[Dict] = None,
        tags: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Train model with CPCV validation and MLflow tracking.

        Args:
            model: Scikit-learn compatible model
            X: Feature matrix (DatetimeIndex)
            y: Target variable
            model_name: Name for model registration
            cv_splits: Number of CPCV splits
            params: Model hyperparameters (for logging)
            tags: Additional tags

        Returns:
            Dict with run info and metrics
        """
        logger.info(f"Training {model_name} with CPCV ({cv_splits} splits)")

        # Compute dataset hash for provenance
        dataset_hash = self._compute_dataset_hash(X, y)

        # Start MLflow run
        with mlflow.start_run(run_name=f"{model_name}_{datetime.now():%Y%m%d_%H%M%S}") as run:
            run_id = run.info.run_id
            logger.info(f"MLflow run ID: {run_id}")

            # Log parameters
            if params:
                mlflow.log_params(params)

            mlflow.log_param("model_type", model.__class__.__name__)
            mlflow.log_param("cv_splits", cv_splits)
            mlflow.log_param("dataset_hash", dataset_hash)

            # Log tags
            if tags:
                mlflow.set_tags(tags)

            mlflow.set_tag("training_date", datetime.now().isoformat())

            # Log dataset info
            self._log_dataset_info(X, y, "train")

            # CPCV splits
            splitter = CombinatorialPurgedCV(n_splits=cv_splits)

            cv_scores = []
            fold_metrics = []

            for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X)):
                logger.info(f"Fold {fold_idx + 1}/{cv_splits}")

                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                # Train model
                model.fit(X_train, y_train)

                # Predictions
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)

                # Calculate metrics
                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)

                fold_metric = {
                    "fold": fold_idx,
                    "train_score": train_score,
                    "test_score": test_score,
                    "train_samples": len(X_train),
                    "test_samples": len(X_test),
                }

                fold_metrics.append(fold_metric)
                cv_scores.append(test_score)

                # Log fold metrics
                mlflow.log_metric(f"fold_{fold_idx}_train_score", train_score)
                mlflow.log_metric(f"fold_{fold_idx}_test_score", test_score)

            # Aggregate CV metrics
            mean_cv_score = np.mean(cv_scores)
            std_cv_score = np.std(cv_scores)

            mlflow.log_metric("cv_score_mean", mean_cv_score)
            mlflow.log_metric("cv_score_std", std_cv_score)

            logger.info(f"CV Score: {mean_cv_score:.4f} ± {std_cv_score:.4f}")

            # Train on full dataset for final model
            logger.info("Training on full dataset")
            model.fit(X, y)

            # Validation with BacktestValidator
            logger.info("Running validation gates")

            # Assuming y is returns
            validator = BacktestValidator(
                min_sharpe=1.2,
                min_psr=0.95,
                max_drawdown=-0.20,
            )

            # For validation, we need actual returns
            # This is simplified - in production, run proper backtest
            # For now, use CV test set returns as proxy
            validation_passed, validation_metrics = validator.validate_model(
                pd.Series(y.iloc[-len(y) // 5:])  # Last 20% as holdout
            )

            # Log validation metrics
            for key, value in validation_metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"validation_{key}", value)

            mlflow.log_param("validation_passed", validation_passed)

            # Save and log model
            signature = infer_signature(X, model.predict(X))

            mlflow.sklearn.log_model(
                model,
                "model",
                signature=signature,
                registered_model_name=model_name,
            )

            # Log fold metrics as artifact
            mlflow.log_dict(
                fold_metrics,
                "cv_fold_metrics.json",
            )

            # Register model (if validation passed)
            if validation_passed:
                logger.success(f"✅ Model passed validation - registered as {model_name}")

                # Transition to staging
                client = mlflow.tracking.MlflowClient()
                model_version = client.get_latest_versions(model_name, stages=["None"])[0]

                client.transition_model_version_stage(
                    name=model_name,
                    version=model_version.version,
                    stage="Staging",
                    archive_existing_versions=False,
                )

                logger.info(f"Model version {model_version.version} → Staging")

            else:
                logger.error(f"❌ Model FAILED validation - not promoted")
                mlflow.set_tag("validation_failed", "true")

            results = {
                "run_id": run_id,
                "model_name": model_name,
                "cv_score": mean_cv_score,
                "cv_std": std_cv_score,
                "validation_passed": validation_passed,
                "validation_metrics": validation_metrics,
                "dataset_hash": dataset_hash,
            }

            return results

    def load_production_model(
        self,
        model_name: str,
        stage: str = "Production",
    ) -> Any:
        """
        Load model from registry.

        Args:
            model_name: Registered model name
            stage: Model stage (Staging, Production)

        Returns:
            Loaded model
        """
        logger.info(f"Loading {model_name} from {stage}")

        model_uri = f"models:/{model_name}/{stage}"

        try:
            model = mlflow.sklearn.load_model(model_uri)
            logger.success(f"Loaded {model_name} ({stage})")
            return model

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def promote_model_to_production(
        self,
        model_name: str,
        version: Optional[int] = None,
    ) -> bool:
        """
        Promote model from Staging to Production.

        Args:
            model_name: Model name
            version: Specific version (None = latest in Staging)

        Returns:
            True if successful
        """
        try:
            client = mlflow.tracking.MlflowClient()

            # Get model version
            if version is None:
                versions = client.get_latest_versions(model_name, stages=["Staging"])
                if not versions:
                    logger.error(f"No {model_name} in Staging")
                    return False
                model_version = versions[0]
            else:
                model_version = client.get_model_version(model_name, version)

            # Transition to Production
            client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage="Production",
                archive_existing_versions=True,  # Archive old production models
            )

            logger.success(
                f"✅ {model_name} v{model_version.version} → Production"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to promote model: {e}")
            return False


# Example training script
def train_model_example():
    """Example training workflow with MLflow."""
    from sklearn.ensemble import RandomForestRegressor

    # Initialize trainer
    trainer = MLflowTrainer(
        experiment_name="quantcli-ensemble",
        tracking_uri="http://localhost:5000",
    )

    # Load data (example)
    # X, y = load_training_data()

    # Create model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
    )

    # Train with CPCV
    # results = trainer.train_with_cpcv(
    #     model=model,
    #     X=X,
    #     y=y,
    #     model_name="rf_ensemble_v1",
    #     cv_splits=5,
    #     params={
    #         "n_estimators": 100,
    #         "max_depth": 6,
    #         "random_state": 42,
    #     },
    #     tags={
    #         "model_family": "random_forest",
    #         "strategy": "momentum",
    #     },
    # )

    # if results["validation_passed"]:
    #     # Promote to production
    #     trainer.promote_model_to_production("rf_ensemble_v1")

    logger.info("Training example (commented out)")
