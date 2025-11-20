"""
Backtesting module for QuantCLI.

Industry-standard backtesting with:
- Advanced transaction cost models
- Short selling support
- Walk-forward analysis
- Monte Carlo simulation
- Comprehensive performance metrics
"""

from .engine import (
    BacktestEngine,
    BacktestResult,
    CostModel,
    FillModel,
    WalkForwardAnalyzer,
    MonteCarloSimulator
)

from .cpcv import (
    CombinatorialPurgedCV,
    BacktestValidator,
    cpcv_indices,
    validate_model_for_production
)

__all__ = [
    'BacktestEngine',
    'BacktestResult',
    'CostModel',
    'FillModel',
    'WalkForwardAnalyzer',
    'MonteCarloSimulator',
    'CombinatorialPurgedCV',
    'BacktestValidator',
    'cpcv_indices',
    'validate_model_for_production'
]
