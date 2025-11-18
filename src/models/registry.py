"""
Model registry for versioning and management.

Tracks:
- Model versions
- Training metadata
- Performance metrics
- Model lineage
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import shutil

from src.core.logging_config import get_logger
from src.core.exceptions import ModelError

logger = get_logger(__name__)


@dataclass
class ModelVersion:
    """
    Model version metadata.

    Attributes:
        model_name: Name of the model
        version: Version number
        created_at: Creation timestamp
        metrics: Performance metrics
        config: Model configuration
        file_path: Path to model file
        metadata: Additional metadata
    """
    model_name: str
    version: str
    created_at: str
    metrics: Dict[str, float]
    config: Dict[str, Any]
    file_path: str
    metadata: Dict[str, Any]


class ModelRegistry:
    """
    Model registry for version control and management.

    Provides:
    - Model versioning
    - Metadata tracking
    - Model promotion (dev -> staging -> production)
    - Model rollback
    """

    def __init__(self, registry_path: Path):
        """
        Initialize model registry.

        Args:
            registry_path: Base path for registry storage
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)

        self.metadata_file = self.registry_path / "registry.json"
        self.models_dir = self.registry_path / "models"
        self.models_dir.mkdir(exist_ok=True)

        # Load existing registry
        self.registry_data = self._load_registry()

        self.logger = logger

    def _load_registry(self) -> Dict[str, Any]:
        """Load registry metadata from disk."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)

        return {
            'models': {},
            'production': {},
            'staging': {},
            'created_at': datetime.now().isoformat()
        }

    def _save_registry(self) -> None:
        """Save registry metadata to disk."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.registry_data, f, indent=2)

    def register_model(
        self,
        model_name: str,
        model_path: Path,
        metrics: Dict[str, float],
        config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        auto_version: bool = True
    ) -> str:
        """
        Register a new model version.

        Args:
            model_name: Name of the model
            model_path: Path to model file
            metrics: Performance metrics
            config: Model configuration
            metadata: Additional metadata
            auto_version: Auto-increment version number

        Returns:
            Version string

        Raises:
            ModelError: If registration fails
        """
        if not model_path.exists():
            raise ModelError(f"Model file not found: {model_path}")

        # Generate version
        if auto_version:
            version = self._get_next_version(model_name)
        else:
            version = metadata.get('version', '1.0.0') if metadata else '1.0.0'

        # Create version directory
        version_dir = self.models_dir / model_name / version
        version_dir.mkdir(parents=True, exist_ok=True)

        # Copy model file
        dest_path = version_dir / model_path.name
        shutil.copy2(model_path, dest_path)

        # Create model version object
        model_version = ModelVersion(
            model_name=model_name,
            version=version,
            created_at=datetime.now().isoformat(),
            metrics=metrics,
            config=config,
            file_path=str(dest_path.relative_to(self.registry_path)),
            metadata=metadata or {}
        )

        # Update registry
        if model_name not in self.registry_data['models']:
            self.registry_data['models'][model_name] = {'versions': []}

        self.registry_data['models'][model_name]['versions'].append(
            asdict(model_version)
        )

        # Save registry
        self._save_registry()

        self.logger.info(
            f"Registered {model_name} v{version} with metrics: {metrics}"
        )

        return version

    def _get_next_version(self, model_name: str) -> str:
        """Generate next version number."""
        if model_name not in self.registry_data['models']:
            return '1.0.0'

        versions = self.registry_data['models'][model_name]['versions']

        if not versions:
            return '1.0.0'

        # Get latest version
        latest = versions[-1]['version']

        # Parse and increment
        major, minor, patch = map(int, latest.split('.'))
        patch += 1

        return f"{major}.{minor}.{patch}"

    def get_model_path(self, model_name: str, version: Optional[str] = None) -> Path:
        """
        Get path to model file.

        Args:
            model_name: Name of model
            version: Specific version (None = latest)

        Returns:
            Path to model file

        Raises:
            ModelError: If model not found
        """
        if model_name not in self.registry_data['models']:
            raise ModelError(f"Model not found: {model_name}")

        versions_list = self.registry_data['models'][model_name]['versions']

        if not versions_list:
            raise ModelError(f"No versions found for {model_name}")

        # Get specific or latest version
        if version:
            model_version = next(
                (v for v in versions_list if v['version'] == version),
                None
            )
            if not model_version:
                raise ModelError(
                    f"Version {version} not found for {model_name}"
                )
        else:
            model_version = versions_list[-1]

        file_path = self.registry_path / model_version['file_path']

        if not file_path.exists():
            raise ModelError(f"Model file not found: {file_path}")

        return file_path

    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all registered models.

        Returns:
            List of model summaries
        """
        models = []

        for model_name, model_data in self.registry_data['models'].items():
            versions = model_data['versions']

            if versions:
                latest = versions[-1]

                models.append({
                    'name': model_name,
                    'latest_version': latest['version'],
                    'created_at': latest['created_at'],
                    'metrics': latest['metrics'],
                    'n_versions': len(versions)
                })

        return models

    def list_versions(self, model_name: str) -> List[ModelVersion]:
        """
        List all versions of a model.

        Args:
            model_name: Name of model

        Returns:
            List of ModelVersion objects
        """
        if model_name not in self.registry_data['models']:
            return []

        versions_data = self.registry_data['models'][model_name]['versions']

        return [
            ModelVersion(**v) for v in versions_data
        ]

    def promote_to_production(self, model_name: str, version: str) -> None:
        """
        Promote model version to production.

        Args:
            model_name: Name of model
            version: Version to promote

        Raises:
            ModelError: If model/version not found
        """
        # Verify model exists
        model_path = self.get_model_path(model_name, version)

        # Update production pointer
        self.registry_data['production'][model_name] = {
            'version': version,
            'promoted_at': datetime.now().isoformat(),
            'file_path': str(model_path.relative_to(self.registry_path))
        }

        self._save_registry()

        self.logger.info(
            f"Promoted {model_name} v{version} to production"
        )

    def get_production_model(self, model_name: str) -> Optional[Path]:
        """
        Get production model path.

        Args:
            model_name: Name of model

        Returns:
            Path to production model, or None if not set
        """
        if model_name not in self.registry_data['production']:
            self.logger.warning(f"No production model set for {model_name}")
            return None

        prod_info = self.registry_data['production'][model_name]
        file_path = self.registry_path / prod_info['file_path']

        if not file_path.exists():
            self.logger.error(f"Production model file not found: {file_path}")
            return None

        return file_path

    def compare_versions(
        self,
        model_name: str,
        metric: str = 'accuracy'
    ) -> Dict[str, float]:
        """
        Compare model versions by metric.

        Args:
            model_name: Name of model
            metric: Metric to compare

        Returns:
            Dictionary mapping version -> metric value
        """
        versions = self.list_versions(model_name)

        comparison = {}

        for version in versions:
            if metric in version.metrics:
                comparison[version.version] = version.metrics[metric]

        return comparison

    def delete_version(self, model_name: str, version: str) -> None:
        """
        Delete a model version.

        Args:
            model_name: Name of model
            version: Version to delete

        Raises:
            ModelError: If trying to delete production version
        """
        # Check if production
        prod_info = self.registry_data['production'].get(model_name)
        if prod_info and prod_info['version'] == version:
            raise ModelError(
                f"Cannot delete production version {version}. "
                f"Promote a different version first."
            )

        # Get model path
        model_path = self.get_model_path(model_name, version)

        # Delete files
        if model_path.parent.exists():
            shutil.rmtree(model_path.parent)

        # Remove from registry
        versions = self.registry_data['models'][model_name]['versions']
        self.registry_data['models'][model_name]['versions'] = [
            v for v in versions if v['version'] != version
        ]

        self._save_registry()

        self.logger.info(f"Deleted {model_name} v{version}")

    def export_metadata(self, output_path: Path) -> None:
        """
        Export registry metadata to file.

        Args:
            output_path: Path to export file
        """
        with open(output_path, 'w') as f:
            json.dump(self.registry_data, f, indent=2)

        self.logger.info(f"Exported metadata to {output_path}")

    def get_best_model(
        self,
        metric: str = 'accuracy',
        higher_is_better: bool = True
    ) -> Optional[Tuple[str, str]]:
        """
        Get best model across all registered models.

        Args:
            metric: Metric to optimize
            higher_is_better: Whether higher metric is better

        Returns:
            Tuple of (model_name, version) or None
        """
        best_model = None
        best_version = None
        best_score = -np.inf if higher_is_better else np.inf

        for model_name in self.registry_data['models']:
            versions = self.list_versions(model_name)

            for version in versions:
                if metric not in version.metrics:
                    continue

                score = version.metrics[metric]

                if higher_is_better:
                    if score > best_score:
                        best_score = score
                        best_model = model_name
                        best_version = version.version
                else:
                    if score < best_score:
                        best_score = score
                        best_model = model_name
                        best_version = version.version

        if best_model:
            self.logger.info(
                f"Best model: {best_model} v{best_version} "
                f"({metric}={best_score:.4f})"
            )

        return (best_model, best_version) if best_model else None
