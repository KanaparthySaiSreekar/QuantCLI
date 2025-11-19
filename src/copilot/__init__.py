"""
QuantCLI AI Copilot Module

Built-in AI assistant powered by open-source LLMs for:
- Signal analysis and explanation
- Portfolio insights and risk assessment
- Market data interpretation
- Configuration optimization suggestions
- Strategy recommendations
- Model performance analysis with explainability
"""

from .service import CopilotService
from .context import ContextProvider
from .explainer import ModelExplainer

__all__ = [
    "CopilotService",
    "ContextProvider",
    "ModelExplainer",
]
