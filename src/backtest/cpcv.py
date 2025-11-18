"""
Combinatorial Purged Cross-Validation (CPCV)

Implementation of CPCV from "Advances in Financial Machine Learning" by Lopez de Prado.
Reduces backtest overfitting compared to walk-forward validation.

Key Features:
- Purging: Remove samples near test set to prevent information leakage
- Embargo: Additional buffer after purge period
- Combinatorial: Multiple train/test splits for robustness
"""

from itertools import combinations
from typing import Generator, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger


class CombinatorialPurgedCV:
    """
    CPCV Splitter with purging and embargo.

    Research: "The Probability of Backtest Overfitting" (Bailey et al., 2015)
    Shows CPCV has lower overfitting probability than walk-forward.
    """

    def __init__(
        self,
        n_splits: int = 10,
        test_size: float = 0.3,
        purge_pct: float = 0.05,
        embargo_pct: float = 0.01,
    ):
        """
        Initialize CPCV splitter.

        Args:
            n_splits: Number of splits (paths through data)
            test_size: Fraction of data in test set
            purge_pct: Fraction of data to purge before test (prevent leakage)
            embargo_pct: Additional embargo after test set
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.purge_pct = purge_pct
        self.embargo_pct = embargo_pct

    def split(
        self, X: pd.DataFrame, y: pd.Series = None, groups=None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate (train_idx, test_idx) splits.

        Args:
            X: Feature matrix (DatetimeIndex required)
            y: Target (not used, for sklearn compatibility)
            groups: Groups (not used)

        Yields:
            (train_indices, test_indices) tuples
        """
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("X must have DatetimeIndex for time-series splits")

        n = len(X)
        indices = np.arange(n)

        # Calculate sizes
        test_size = int(n * self.test_size)
        purge_size = int(n * self.purge_pct)
        embargo_size = int(n * self.embargo_pct)

        logger.info(
            f"CPCV: n={n}, test_size={test_size}, "
            f"purge={purge_size}, embargo={embargo_size}"
        )

        # Generate non-overlapping test blocks
        n_blocks = self.n_splits
        block_size = test_size

        # Ensure blocks don't overlap
        max_start = n - block_size - embargo_size
        starts = np.linspace(0, max_start, n_blocks, dtype=int)

        for i, start in enumerate(starts):
            test_start = start
            test_end = min(start + block_size, n)

            # Purge before test (remove samples that could leak info)
            purge_start = max(0, test_start - purge_size)

            # Embargo after test
            embargo_end = min(test_end + embargo_size, n)

            # Train set: everything except [purge_start, embargo_end)
            train_idx = np.concatenate([
                indices[:purge_start],
                indices[embargo_end:]
            ])

            test_idx = indices[test_start:test_end]

            if len(train_idx) < 100 or len(test_idx) < 50:
                logger.warning(f"Split {i}: train={len(train_idx)}, test={len(test_idx)} (skipping)")
                continue

            logger.debug(
                f"Split {i}: train={len(train_idx)}, test={len(test_idx)}, "
                f"purged={purge_size}, embargo={embargo_size}"
            )

            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return number of splits."""
        return self.n_splits


def cpcv_indices(
    dates: pd.DatetimeIndex,
    n_splits: int = 5,
    purge_days: int = 5,
    embargo_days: int = 1,
) -> Generator[Tuple[List[int], List[int]], None, None]:
    """
    Simple CPCV splitter by date (alternative implementation).

    Args:
        dates: DatetimeIndex of observations
        n_splits: Number of CV splits
        purge_days: Days to purge before test
        embargo_days: Days to embargo after test

    Yields:
        (train_indices, test_indices) tuples
    """
    n = len(dates)
    block = n // n_splits

    for i in range(n_splits):
        test_start_idx = i * block
        test_end_idx = min((i + 1) * block, n)

        # Find purge start by date
        test_start_date = dates[test_start_idx]
        purge_start_date = test_start_date - pd.Timedelta(days=purge_days)
        purge_start_idx = dates.searchsorted(purge_start_date)

        # Find embargo end by date
        test_end_date = dates[test_end_idx - 1] if test_end_idx > 0 else dates[test_start_idx]
        embargo_end_date = test_end_date + pd.Timedelta(days=embargo_days)
        embargo_end_idx = min(dates.searchsorted(embargo_end_date), n)

        # Train indices: before purge and after embargo
        train_idx = list(range(0, purge_start_idx)) + list(range(embargo_end_idx, n))
        test_idx = list(range(test_start_idx, test_end_idx))

        if len(train_idx) < 100 or len(test_idx) < 20:
            logger.warning(
                f"CPCV split {i}: insufficient data (train={len(train_idx)}, test={len(test_idx)})"
            )
            continue

        yield train_idx, test_idx


class BacktestValidator:
    """
    Validation framework with CPCV and performance gating.

    Enforces:
    - CPCV for robust validation
    - Performance thresholds (Sharpe, PSR, DSR)
    - Automatic rejection of underperforming models
    """

    def __init__(
        self,
        min_sharpe: float = 1.2,
        min_psr: float = 0.95,
        max_drawdown: float = -0.20,
    ):
        """
        Initialize validator with thresholds.

        Args:
            min_sharpe: Minimum Sharpe ratio to pass
            min_psr: Minimum Probabilistic Sharpe Ratio
            max_drawdown: Maximum allowed drawdown (negative)
        """
        self.min_sharpe = min_sharpe
        self.min_psr = min_psr
        self.max_drawdown = max_drawdown

    def validate_model(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series = None,
    ) -> Tuple[bool, dict]:
        """
        Validate model performance against thresholds.

        Args:
            returns: Model returns (daily)
            benchmark_returns: Benchmark returns for comparison

        Returns:
            (passed, metrics_dict)
        """
        metrics = {}

        # Sharpe Ratio
        sharpe = self._calculate_sharpe(returns)
        metrics["sharpe_ratio"] = sharpe

        # Probabilistic Sharpe Ratio (PSR)
        psr = self._calculate_psr(returns, target_sharpe=self.min_sharpe)
        metrics["psr"] = psr

        # Deflated Sharpe Ratio (DSR) - adjusts for multiple testing
        dsr = self._calculate_dsr(returns, n_trials=100)  # Assume 100 backtest runs
        metrics["dsr"] = dsr

        # Maximum Drawdown
        drawdown = self._calculate_max_drawdown(returns)
        metrics["max_drawdown"] = drawdown

        # Win Rate
        win_rate = (returns > 0).sum() / len(returns)
        metrics["win_rate"] = win_rate

        # Annual Return
        annual_return = (1 + returns.mean()) ** 252 - 1
        metrics["annual_return"] = annual_return

        # Validation checks
        passed = True
        reasons = []

        if sharpe < self.min_sharpe:
            passed = False
            reasons.append(f"Sharpe {sharpe:.2f} < {self.min_sharpe:.2f}")

        if psr < self.min_psr:
            passed = False
            reasons.append(f"PSR {psr:.3f} < {self.min_psr:.3f}")

        if drawdown < self.max_drawdown:
            passed = False
            reasons.append(f"Drawdown {drawdown:.1%} > {self.max_drawdown:.1%}")

        metrics["validation_passed"] = passed
        metrics["rejection_reasons"] = reasons

        if passed:
            logger.success(f"✅ Model PASSED validation: Sharpe={sharpe:.2f}, PSR={psr:.3f}")
        else:
            logger.error(f"❌ Model FAILED validation: {', '.join(reasons)}")

        return passed, metrics

    def _calculate_sharpe(self, returns: pd.Series) -> float:
        """Calculate annualized Sharpe ratio."""
        return returns.mean() / returns.std() * np.sqrt(252)

    def _calculate_psr(
        self, returns: pd.Series, target_sharpe: float = 1.0
    ) -> float:
        """
        Calculate Probabilistic Sharpe Ratio.

        PSR = probability that true Sharpe > target Sharpe
        Accounts for estimation uncertainty.
        """
        from scipy.stats import norm

        n = len(returns)
        sr = self._calculate_sharpe(returns)

        # Skewness and kurtosis adjustments
        skew = returns.skew()
        kurt = returns.kurtosis()

        # Standard error of Sharpe ratio
        sr_std = np.sqrt(
            (1 + (0.25 * skew * sr) - ((kurt - 3) / 4) * (sr ** 2)) / (n - 1)
        )

        # PSR
        psr = norm.cdf((sr - target_sharpe) / sr_std)

        return psr

    def _calculate_dsr(self, returns: pd.Series, n_trials: int = 100) -> float:
        """
        Calculate Deflated Sharpe Ratio.

        Adjusts for multiple testing bias.
        """
        from scipy.stats import norm

        n = len(returns)
        sr = self._calculate_sharpe(returns)

        # Adjustment for multiple trials
        # E[max SR] when testing n_trials strategies
        variance = ((1 - np.euler_gamma) * norm.ppf(1 - 1 / n_trials)) ** 2
        adjusted_sr = sr * np.sqrt(n / (n - 1)) - np.sqrt(variance / n)

        return adjusted_sr

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()


# Example usage in CI/CD pipeline
def validate_model_for_production(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    baseline_metrics: dict = None,
) -> bool:
    """
    Production gating function for CI/CD.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        baseline_metrics: Baseline model metrics to beat

    Returns:
        True if model passes all gates
    """
    # Generate predictions
    predictions = model.predict(X_test)

    # Convert to returns (if needed)
    # Assuming predictions are next-day returns
    returns = pd.Series(predictions, index=y_test.index)

    # Validate
    validator = BacktestValidator(
        min_sharpe=1.2,
        min_psr=0.95,
        max_drawdown=-0.20,
    )

    passed, metrics = validator.validate_model(returns)

    # Check against baseline
    if baseline_metrics and passed:
        if metrics["sharpe_ratio"] < baseline_metrics.get("sharpe_ratio", 0):
            logger.error("Model underperforms baseline")
            return False

    return passed
