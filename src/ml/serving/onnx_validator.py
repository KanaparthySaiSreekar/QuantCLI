"""
ONNX Model Conversion, Quantization, and Validation

Ensures models are production-ready with:
- ONNX conversion for cross-platform inference
- INT8 quantization for 2-4x speedup
- Parity testing (<1% accuracy difference)
- Performance benchmarking
"""

import time
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import onnx
import onnxruntime as ort
from loguru import logger
from onnxruntime.quantization import QuantType, quantize_dynamic
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


class ONNXConverter:
    """Convert sklearn/XGBoost/LightGBM models to ONNX."""

    def __init__(self, opset_version: int = 17):
        """
        Initialize ONNX converter.

        Args:
            opset_version: ONNX opset version (17 recommended)
        """
        self.opset_version = opset_version

    def convert_sklearn_model(
        self,
        model,
        initial_types: list,
        output_path: Path,
    ) -> bool:
        """
        Convert sklearn model to ONNX.

        Args:
            model: Trained sklearn model
            initial_types: Input types (e.g., [('float_input', FloatTensorType([None, 50]))])
            output_path: Path to save ONNX model

        Returns:
            True if successful
        """
        try:
            logger.info(f"Converting sklearn model to ONNX (opset={self.opset_version})")

            onnx_model = convert_sklearn(
                model,
                initial_types=initial_types,
                target_opset=self.opset_version,
            )

            # Save
            onnx.save_model(onnx_model, str(output_path))
            logger.success(f"ONNX model saved to {output_path}")

            return True

        except Exception as e:
            logger.error(f"ONNX conversion failed: {e}")
            return False

    def convert_xgboost_model(
        self,
        model,
        n_features: int,
        output_path: Path,
    ) -> bool:
        """
        Convert XGBoost model to ONNX.

        Args:
            model: Trained XGBoost model
            n_features: Number of input features
            output_path: Path to save ONNX model

        Returns:
            True if successful
        """
        try:
            import xgboost as xgb

            logger.info("Converting XGBoost model to ONNX")

            initial_types = [("float_input", FloatTensorType([None, n_features]))]

            onnx_model = convert_sklearn(
                model,
                initial_types=initial_types,
                target_opset=self.opset_version,
            )

            onnx.save_model(onnx_model, str(output_path))
            logger.success(f"ONNX model saved to {output_path}")

            return True

        except Exception as e:
            logger.error(f"XGBoost ONNX conversion failed: {e}")
            return False


class ONNXQuantizer:
    """Quantize ONNX models to INT8 for faster inference."""

    def __init__(
        self,
        per_channel: bool = True,
        symmetric: bool = True,
    ):
        """
        Initialize quantizer.

        Args:
            per_channel: Per-channel quantization (better accuracy)
            symmetric: Symmetric quantization
        """
        self.per_channel = per_channel
        self.symmetric = symmetric

    def quantize_dynamic(
        self,
        input_path: Path,
        output_path: Path,
        weight_type: QuantType = QuantType.QInt8,
    ) -> bool:
        """
        Apply dynamic INT8 quantization.

        Args:
            input_path: Path to FP32 ONNX model
            output_path: Path to save INT8 model
            weight_type: Quantization type (QInt8 recommended)

        Returns:
            True if successful
        """
        try:
            logger.info(f"Quantizing {input_path} to INT8")

            quantize_dynamic(
                model_input=str(input_path),
                model_output=str(output_path),
                weight_type=weight_type,
                per_channel=self.per_channel,
                reduce_range=False,
                optimize_model=True,
            )

            # Check file size reduction
            original_size = input_path.stat().st_size / (1024 * 1024)  # MB
            quantized_size = output_path.stat().st_size / (1024 * 1024)  # MB
            reduction = (1 - quantized_size / original_size) * 100

            logger.success(
                f"Quantized model saved: {original_size:.1f}MB → {quantized_size:.1f}MB "
                f"({reduction:.1f}% reduction)"
            )

            return True

        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            return False


class ONNXParityTester:
    """Test parity between original and ONNX models."""

    def __init__(self, atol: float = 1e-3, rtol: float = 1e-2):
        """
        Initialize parity tester.

        Args:
            atol: Absolute tolerance for np.allclose
            rtol: Relative tolerance
        """
        self.atol = atol
        self.rtol = rtol

    def test_parity(
        self,
        original_model,
        onnx_path: Path,
        X_sample: np.ndarray,
        model_type: str = "sklearn",
    ) -> Tuple[bool, Dict]:
        """
        Test parity between original and ONNX model.

        Args:
            original_model: Original trained model
            onnx_path: Path to ONNX model
            X_sample: Sample input data for testing
            model_type: Type of model (sklearn, xgboost, lightgbm)

        Returns:
            (passed, metrics_dict)
        """
        try:
            logger.info(f"Testing parity for {onnx_path}")

            # Original predictions
            if model_type in ["sklearn", "xgboost", "lightgbm"]:
                pred_original = original_model.predict(X_sample)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            # ONNX predictions
            sess = ort.InferenceSession(
                str(onnx_path),
                providers=["CPUExecutionProvider"],
            )

            input_name = sess.get_inputs()[0].name
            X_sample_float = X_sample.astype(np.float32)
            pred_onnx = sess.run(None, {input_name: X_sample_float})[0]

            # Flatten if needed
            if len(pred_onnx.shape) > 1:
                pred_onnx = pred_onnx.flatten()

            # Calculate metrics
            mae = np.mean(np.abs(pred_original - pred_onnx))
            rmse = np.sqrt(np.mean((pred_original - pred_onnx) ** 2))
            max_diff = np.max(np.abs(pred_original - pred_onnx))
            rel_error = mae / (np.mean(np.abs(pred_original)) + 1e-8)

            # Parity check
            passed = np.allclose(pred_original, pred_onnx, atol=self.atol, rtol=self.rtol)

            metrics = {
                "passed": passed,
                "mae": float(mae),
                "rmse": float(rmse),
                "max_diff": float(max_diff),
                "relative_error_pct": float(rel_error * 100),
            }

            if passed:
                logger.success(
                    f"✅ Parity test PASSED: MAE={mae:.6f}, "
                    f"RMSE={rmse:.6f}, MaxDiff={max_diff:.6f}"
                )
            else:
                logger.error(
                    f"❌ Parity test FAILED: MAE={mae:.6f}, "
                    f"RMSE={rmse:.6f}, MaxDiff={max_diff:.6f}"
                )

            return passed, metrics

        except Exception as e:
            logger.error(f"Parity test failed with exception: {e}")
            return False, {"error": str(e)}

    def benchmark_performance(
        self,
        original_model,
        onnx_path: Path,
        X_sample: np.ndarray,
        n_runs: int = 1000,
    ) -> Dict:
        """
        Benchmark inference performance.

        Args:
            original_model: Original model
            onnx_path: ONNX model path
            X_sample: Sample input
            n_runs: Number of inference runs

        Returns:
            Performance metrics dict
        """
        logger.info(f"Benchmarking performance ({n_runs} runs)")

        # Benchmark original model
        start = time.perf_counter()
        for _ in range(n_runs):
            _ = original_model.predict(X_sample)
        original_time = (time.perf_counter() - start) / n_runs * 1000  # ms

        # Benchmark ONNX model
        sess = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        input_name = sess.get_inputs()[0].name
        X_sample_float = X_sample.astype(np.float32)

        start = time.perf_counter()
        for _ in range(n_runs):
            _ = sess.run(None, {input_name: X_sample_float})
        onnx_time = (time.perf_counter() - start) / n_runs * 1000  # ms

        speedup = original_time / onnx_time

        metrics = {
            "original_latency_ms": float(original_time),
            "onnx_latency_ms": float(onnx_time),
            "speedup": float(speedup),
        }

        logger.info(
            f"Performance: Original={original_time:.2f}ms, "
            f"ONNX={onnx_time:.2f}ms, Speedup={speedup:.2f}x"
        )

        return metrics


def convert_and_validate_pipeline(
    model,
    model_name: str,
    X_sample: np.ndarray,
    output_dir: Path,
    model_type: str = "sklearn",
) -> Dict:
    """
    Complete pipeline: Convert → Quantize → Validate.

    Args:
        model: Trained model
        model_name: Model name for file naming
        X_sample: Sample data for validation
        output_dir: Directory to save models
        model_type: Type of model

    Returns:
        Dict with all metrics and paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "model_name": model_name,
        "success": False,
    }

    # Paths
    fp32_path = output_dir / f"{model_name}_fp32.onnx"
    int8_path = output_dir / f"{model_name}_int8.onnx"

    # Step 1: Convert to ONNX FP32
    logger.info("Step 1: Converting to ONNX FP32")
    converter = ONNXConverter()

    n_features = X_sample.shape[1]
    initial_types = [("float_input", FloatTensorType([None, n_features]))]

    if not converter.convert_sklearn_model(model, initial_types, fp32_path):
        results["error"] = "FP32 conversion failed"
        return results

    results["fp32_model_path"] = str(fp32_path)

    # Step 2: Test FP32 parity
    logger.info("Step 2: Testing FP32 parity")
    tester = ONNXParityTester()

    fp32_passed, fp32_metrics = tester.test_parity(
        model, fp32_path, X_sample, model_type
    )

    if not fp32_passed:
        results["error"] = "FP32 parity test failed"
        results["fp32_metrics"] = fp32_metrics
        return results

    results["fp32_parity"] = fp32_metrics

    # Step 3: Quantize to INT8
    logger.info("Step 3: Quantizing to INT8")
    quantizer = ONNXQuantizer()

    if not quantizer.quantize_dynamic(fp32_path, int8_path):
        results["error"] = "INT8 quantization failed"
        return results

    results["int8_model_path"] = str(int8_path)

    # Step 4: Test INT8 parity (relaxed tolerance)
    logger.info("Step 4: Testing INT8 parity")
    tester_relaxed = ONNXParityTester(atol=1e-2, rtol=5e-2)  # Relaxed for INT8

    int8_passed, int8_metrics = tester_relaxed.test_parity(
        model, int8_path, X_sample, model_type
    )

    results["int8_parity"] = int8_metrics

    # Warn if INT8 accuracy loss > 1%
    if int8_metrics.get("relative_error_pct", 0) > 1.0:
        logger.warning(
            f"INT8 accuracy loss: {int8_metrics['relative_error_pct']:.2f}% (>1%)"
        )

    # Step 5: Benchmark performance
    logger.info("Step 5: Benchmarking performance")

    fp32_perf = tester.benchmark_performance(model, fp32_path, X_sample, n_runs=1000)
    int8_perf = tester.benchmark_performance(model, int8_path, X_sample, n_runs=1000)

    results["fp32_performance"] = fp32_perf
    results["int8_performance"] = int8_perf

    # Summary
    results["success"] = True
    results["summary"] = {
        "fp32_speedup": fp32_perf["speedup"],
        "int8_speedup": int8_perf["speedup"],
        "int8_accuracy_loss_pct": int8_metrics.get("relative_error_pct", 0),
        "recommended_model": "int8" if int8_metrics.get("relative_error_pct", 0) < 1.0 else "fp32",
    }

    logger.success(
        f"✅ Pipeline complete: INT8 speedup={int8_perf['speedup']:.2f}x, "
        f"accuracy loss={int8_metrics.get('relative_error_pct', 0):.2f}%"
    )

    return results


# Example usage for CI/CD
def validate_onnx_for_ci(
    model_path: str,
    onnx_path: str,
    test_data_path: str,
) -> bool:
    """
    CI/CD validation function.

    Returns:
        True if ONNX model passes all checks
    """
    # Load models
    model = joblib.load(model_path)
    X_test = np.load(test_data_path)

    # Test parity
    tester = ONNXParityTester(atol=1e-3, rtol=1e-2)
    passed, metrics = tester.test_parity(model, Path(onnx_path), X_test)

    if not passed:
        logger.error("ONNX parity test failed - blocking deployment")
        return False

    # Check accuracy loss
    if metrics.get("relative_error_pct", 0) > 1.0:
        logger.error(f"Accuracy loss {metrics['relative_error_pct']:.2f}% > 1%")
        return False

    logger.success("ONNX validation passed")
    return True
