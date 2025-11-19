"""
Model Explainability using SHAP (SHapley Additive exPlanations).

Provides interpretable explanations for ML model predictions.
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
from loguru import logger

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not installed. Install with: pip install shap")


class ModelExplainer:
    """
    Provides model explainability using SHAP values.

    Features:
    - Feature importance analysis
    - Individual prediction explanations
    - Feature interaction detection
    - Visualization-ready outputs
    """

    def __init__(self):
        """Initialize the model explainer."""
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available - explainability features limited")

        self.explainer = None
        self.shap_values = None
        self.feature_names = None

    def create_explainer(
        self,
        model: Any,
        background_data: Union[np.ndarray, pd.DataFrame],
        model_type: str = "tree",
    ):
        """
        Create a SHAP explainer for the model.

        Args:
            model: The ML model to explain
            background_data: Background data for SHAP
            model_type: Type of model ("tree", "linear", "deep", "kernel")
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required for model explanation")

        try:
            if model_type == "tree":
                # For tree-based models (XGBoost, LightGBM, CatBoost)
                self.explainer = shap.TreeExplainer(model)
                logger.info("Created TreeExplainer")

            elif model_type == "linear":
                # For linear models
                self.explainer = shap.LinearExplainer(model, background_data)
                logger.info("Created LinearExplainer")

            elif model_type == "deep":
                # For deep learning models
                self.explainer = shap.DeepExplainer(model, background_data)
                logger.info("Created DeepExplainer")

            else:
                # Kernel explainer works for any model but is slower
                self.explainer = shap.KernelExplainer(
                    model.predict,
                    shap.sample(background_data, 100)
                )
                logger.info("Created KernelExplainer")

            if isinstance(background_data, pd.DataFrame):
                self.feature_names = background_data.columns.tolist()
            else:
                self.feature_names = [f"feature_{i}" for i in range(background_data.shape[1])]

        except Exception as e:
            logger.error(f"Failed to create explainer: {e}")
            raise

    def explain_prediction(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        prediction: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Explain a single prediction or batch of predictions.

        Args:
            X: Input features (single sample or batch)
            prediction: The model's prediction (optional)

        Returns:
            Explanation dictionary with SHAP values and interpretation
        """
        if not SHAP_AVAILABLE or self.explainer is None:
            return {"error": "SHAP explainer not available"}

        try:
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(X)

            # Handle single prediction case
            if len(X.shape) == 1 or X.shape[0] == 1:
                if len(X.shape) == 1:
                    X = X.reshape(1, -1)
                    if isinstance(shap_values, list):
                        shap_values = shap_values[0]
                    if len(shap_values.shape) > 1:
                        shap_values = shap_values[0]

                # Get feature contributions
                feature_contributions = {}
                for i, feature in enumerate(self.feature_names):
                    feature_contributions[feature] = float(shap_values[i])

                # Sort by absolute contribution
                sorted_contributions = sorted(
                    feature_contributions.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )

                # Get top positive and negative contributors
                top_positive = [
                    (feat, val) for feat, val in sorted_contributions if val > 0
                ][:5]
                top_negative = [
                    (feat, val) for feat, val in sorted_contributions if val < 0
                ][:5]

                explanation = {
                    "prediction": prediction,
                    "base_value": float(self.explainer.expected_value) if hasattr(
                        self.explainer, 'expected_value'
                    ) else None,
                    "feature_contributions": feature_contributions,
                    "top_positive_contributors": [
                        {"feature": f, "contribution": v} for f, v in top_positive
                    ],
                    "top_negative_contributors": [
                        {"feature": f, "contribution": v} for f, v in top_negative
                    ],
                    "total_impact": float(sum(shap_values)),
                }

                return explanation

            else:
                # Batch explanation
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]

                return {
                    "batch_size": X.shape[0],
                    "shap_values_shape": shap_values.shape,
                    "mean_abs_shap": {
                        feat: float(np.abs(shap_values[:, i]).mean())
                        for i, feat in enumerate(self.feature_names)
                    },
                }

        except Exception as e:
            logger.error(f"Failed to explain prediction: {e}")
            return {"error": str(e)}

    def get_feature_importance(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        method: str = "mean_abs",
    ) -> Dict[str, float]:
        """
        Get global feature importance.

        Args:
            X: Input features to analyze
            method: Method to compute importance ("mean_abs", "mean")

        Returns:
            Feature importance dictionary
        """
        if not SHAP_AVAILABLE or self.explainer is None:
            return {"error": "SHAP explainer not available"}

        try:
            shap_values = self.explainer.shap_values(X)

            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            if method == "mean_abs":
                importance = np.abs(shap_values).mean(axis=0)
            else:
                importance = shap_values.mean(axis=0)

            feature_importance = {
                feat: float(importance[i])
                for i, feat in enumerate(self.feature_names)
            }

            # Sort by importance
            sorted_importance = dict(
                sorted(
                    feature_importance.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )
            )

            return sorted_importance

        except Exception as e:
            logger.error(f"Failed to get feature importance: {e}")
            return {"error": str(e)}

    def detect_interactions(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Detect feature interactions.

        Args:
            X: Input features
            top_k: Number of top interactions to return

        Returns:
            List of feature interaction dictionaries
        """
        if not SHAP_AVAILABLE or self.explainer is None:
            return [{"error": "SHAP explainer not available"}]

        try:
            # Calculate interaction values if supported
            if hasattr(self.explainer, 'shap_interaction_values'):
                interaction_values = self.explainer.shap_interaction_values(X)

                # Average interaction strengths
                avg_interactions = np.abs(interaction_values).mean(axis=0)

                # Find top interactions
                interactions = []
                n_features = len(self.feature_names)

                for i in range(n_features):
                    for j in range(i + 1, n_features):
                        strength = float(avg_interactions[i, j])
                        interactions.append({
                            "feature_1": self.feature_names[i],
                            "feature_2": self.feature_names[j],
                            "interaction_strength": strength,
                        })

                # Sort by strength
                interactions.sort(key=lambda x: x["interaction_strength"], reverse=True)

                return interactions[:top_k]

            else:
                logger.warning("Interaction values not supported for this explainer type")
                return [{"warning": "Interaction detection not supported"}]

        except Exception as e:
            logger.error(f"Failed to detect interactions: {e}")
            return [{"error": str(e)}]

    def explain_ensemble(
        self,
        models: List[Any],
        model_names: List[str],
        X: Union[np.ndarray, pd.DataFrame],
        background_data: Union[np.ndarray, pd.DataFrame],
    ) -> Dict[str, Any]:
        """
        Explain predictions from an ensemble of models.

        Args:
            models: List of models in ensemble
            model_names: Names of the models
            X: Input features
            background_data: Background data for SHAP

        Returns:
            Ensemble explanation dictionary
        """
        if not SHAP_AVAILABLE:
            return {"error": "SHAP not available"}

        try:
            ensemble_explanation = {
                "models": {},
                "consensus_features": {},
            }

            # Explain each model
            for model, name in zip(models, model_names):
                try:
                    # Create explainer for this model
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X)

                    if isinstance(shap_values, list):
                        shap_values = shap_values[0]

                    # Get mean absolute SHAP values
                    mean_abs_shap = np.abs(shap_values).mean(axis=0)

                    ensemble_explanation["models"][name] = {
                        "feature_importance": {
                            feat: float(mean_abs_shap[i])
                            for i, feat in enumerate(self.feature_names)
                        }
                    }

                except Exception as e:
                    logger.warning(f"Could not explain {name}: {e}")
                    ensemble_explanation["models"][name] = {"error": str(e)}

            # Find consensus important features
            all_importances = []
            for model_info in ensemble_explanation["models"].values():
                if "feature_importance" in model_info:
                    all_importances.append(model_info["feature_importance"])

            if all_importances:
                # Average importance across models
                for feature in self.feature_names:
                    values = [imp.get(feature, 0) for imp in all_importances]
                    ensemble_explanation["consensus_features"][feature] = {
                        "mean_importance": float(np.mean(values)),
                        "std_importance": float(np.std(values)),
                        "agreement": float(np.std(values) / (np.mean(values) + 1e-10)),
                    }

                # Sort by mean importance
                ensemble_explanation["consensus_features"] = dict(
                    sorted(
                        ensemble_explanation["consensus_features"].items(),
                        key=lambda x: x[1]["mean_importance"],
                        reverse=True
                    )
                )

            return ensemble_explanation

        except Exception as e:
            logger.error(f"Failed to explain ensemble: {e}")
            return {"error": str(e)}

    def generate_summary(
        self,
        shap_explanation: Dict[str, Any],
        prediction: float,
        threshold: float = 0.5,
    ) -> str:
        """
        Generate a natural language summary of SHAP explanation.

        Args:
            shap_explanation: SHAP explanation dictionary
            prediction: Model prediction
            threshold: Threshold for prediction interpretation

        Returns:
            Natural language summary
        """
        try:
            if "error" in shap_explanation:
                return f"Could not generate explanation: {shap_explanation['error']}"

            summary_parts = []

            # Prediction interpretation
            if prediction is not None:
                pred_str = f"Prediction: {prediction:.4f}"
                if prediction > threshold:
                    pred_str += " (BUY signal)"
                else:
                    pred_str += " (SELL/HOLD signal)"
                summary_parts.append(pred_str)

            # Top positive contributors
            if "top_positive_contributors" in shap_explanation:
                positive = shap_explanation["top_positive_contributors"]
                if positive:
                    summary_parts.append("\nTop factors supporting this prediction:")
                    for item in positive[:3]:
                        summary_parts.append(
                            f"  - {item['feature']}: {item['contribution']:+.4f}"
                        )

            # Top negative contributors
            if "top_negative_contributors" in shap_explanation:
                negative = shap_explanation["top_negative_contributors"]
                if negative:
                    summary_parts.append("\nTop factors against this prediction:")
                    for item in negative[:3]:
                        summary_parts.append(
                            f"  - {item['feature']}: {item['contribution']:+.4f}"
                        )

            # Base value context
            if "base_value" in shap_explanation and shap_explanation["base_value"]:
                summary_parts.append(
                    f"\nBase prediction (no features): {shap_explanation['base_value']:.4f}"
                )

            return "\n".join(summary_parts)

        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return f"Error generating summary: {e}"
