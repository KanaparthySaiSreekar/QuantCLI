# AI Copilot Module

Built-in AI assistant for QuantCLI powered by open-source LLMs.

## Quick Start

```bash
# Interactive chat
python scripts/copilot.py chat

# Ask a question
python scripts/copilot.py ask "What's my portfolio status?"

# Analyze signals
python scripts/copilot.py analyze-signal AAPL

# Portfolio insights
python scripts/copilot.py explain-portfolio
```

## Module Structure

```
src/copilot/
├── __init__.py           # Module exports
├── service.py            # Core CopilotService (LLM integration)
├── context.py            # ContextProvider (trading data access)
├── explainer.py          # ModelExplainer (SHAP-based interpretability)
└── prompts.py            # Prompt templates for different tasks
```

## Components

### CopilotService (`service.py`)

Core AI service with:
- LLM model loading and inference
- Response caching for performance
- Multiple LLM backend support
- Memory-efficient quantization
- GPU/CPU inference

**Key Methods:**
- `generate_response()` - Generate LLM response
- `analyze_signal()` - Analyze trading signals
- `analyze_portfolio()` - Portfolio insights
- `analyze_model_performance()` - ML model analysis
- `interpret_market()` - Market conditions
- `ask()` - General questions

### ContextProvider (`context.py`)

Provides trading context from the system:
- Portfolio summary and positions
- Recent trades and signals
- Performance metrics
- Market data
- Model information
- Configuration settings

**Key Methods:**
- `get_portfolio_summary()` - Portfolio overview
- `get_current_positions()` - Active positions
- `get_recent_trades()` - Trade history
- `get_recent_signals()` - Signal history
- `get_performance_metrics()` - Performance stats
- `get_market_data()` - Market prices and indicators
- `get_full_context()` - Comprehensive context

### ModelExplainer (`explainer.py`)

SHAP-based model interpretability:
- Feature importance analysis
- Individual prediction explanations
- Feature interaction detection
- Ensemble model analysis
- Natural language summaries

**Key Methods:**
- `create_explainer()` - Initialize SHAP explainer
- `explain_prediction()` - Explain single prediction
- `get_feature_importance()` - Global importance
- `detect_interactions()` - Feature interactions
- `explain_ensemble()` - Multi-model analysis
- `generate_summary()` - Natural language output

### Prompts (`prompts.py`)

Domain-specific prompt templates:
- `SIGNAL_ANALYSIS_PROMPT` - Trading signal analysis
- `PORTFOLIO_ANALYSIS_PROMPT` - Portfolio assessment
- `MODEL_PERFORMANCE_PROMPT` - ML model insights
- `MARKET_INTERPRETATION_PROMPT` - Market conditions
- `STRATEGY_RECOMMENDATION_PROMPT` - Strategy suggestions
- `CONFIG_OPTIMIZATION_PROMPT` - Configuration tuning
- `GENERAL_QUERY_PROMPT` - General questions
- `EXPLAIN_PREDICTION_PROMPT` - Prediction explanation
- `BACKTEST_ANALYSIS_PROMPT` - Backtest results

## Usage Examples

### Python API

```python
from src.copilot import CopilotService, ContextProvider, ModelExplainer

# Initialize services
copilot = CopilotService()
context = ContextProvider()
explainer = ModelExplainer()

# Ask a question
response = copilot.ask("What's my best performing position?")

# Analyze a signal
analysis = copilot.analyze_signal(
    signal_data={"symbol": "AAPL", "direction": "BUY"},
    market_context={"price": 150.0},
    features={"rsi": 65, "macd": 0.5}
)

# Get portfolio insights
portfolio = context.get_portfolio_summary()
positions = context.get_current_positions()
insights = copilot.analyze_portfolio(
    portfolio_data=portfolio,
    positions=positions,
    performance_metrics=context.get_performance_metrics(),
    market_context={}
)

# Explain model predictions
explainer.create_explainer(model, X_train, model_type="tree")
explanation = explainer.explain_prediction(X_test[0], prediction=0.75)
summary = explainer.generate_summary(explanation, 0.75)
```

### CLI

```bash
# Interactive mode
python scripts/copilot.py chat

# Ask questions
python scripts/copilot.py ask "How is my strategy performing?"
python scripts/copilot.py ask "Should I buy AAPL?" --symbol AAPL --context

# Analyze signals
python scripts/copilot.py analyze-signal AAPL --hours 48

# Portfolio analysis
python scripts/copilot.py explain-portfolio --days 30

# Market insights
python scripts/copilot.py market-insight TSLA --days 30

# Model management
python scripts/copilot.py load-model
python scripts/copilot.py status
python scripts/copilot.py clear-cache
```

## Supported Models

Default: `microsoft/Phi-3-mini-4k-instruct` (3.8B)

Alternatives:
- `meta-llama/Llama-3.2-3B-Instruct` (fast, 3B)
- `mistralai/Mistral-7B-Instruct-v0.2` (capable, 7B)
- `HuggingFaceH4/zephyr-7b-beta` (helpful, 7B)

Change model:
```python
copilot.load_model(model_name="meta-llama/Llama-3.2-3B-Instruct")
```

## Features

✅ **Local LLM** - No API costs, runs offline
✅ **Context-aware** - Understands your portfolio and trading state
✅ **Explainable** - SHAP-based model interpretability
✅ **Fast** - Response caching and GPU acceleration
✅ **Private** - All data stays local
✅ **Open-source** - Fully auditable

## Dependencies

- `transformers` - HuggingFace transformers library
- `torch` - PyTorch for model inference
- `accelerate` - Fast model loading
- `bitsandbytes` - 4-bit quantization
- `shap` - Model explainability
- `click` - CLI framework
- `rich` - Beautiful terminal output

## Architecture

```
User Query
    ↓
CLI (scripts/copilot.py)
    ↓
CopilotService
    ├─→ ContextProvider → Database/Config
    ├─→ ModelExplainer → SHAP
    └─→ LLM (HuggingFace) → Response
    ↓
Rich Terminal Output
```

## Performance

**Model Loading:**
- First time: 10-30 seconds (downloads model)
- Subsequent: 5-10 seconds (loads from cache)
- With `load_model` pre-run: <1 second

**Inference Speed:**
- CPU (3B model): 2-5 seconds per response
- GPU (3B model): 0.5-1 second per response
- GPU (7B model): 1-2 seconds per response

**Memory Usage:**
- 3B model (4-bit): ~3 GB
- 7B model (4-bit): ~5 GB
- Without quantization: 2-4x more

## Configuration

Models are configured in `CopilotService._get_model_name()`.

To change default model, edit `service.py:104`:

```python
def _get_model_name(self) -> str:
    return "your-preferred-model-name"
```

Or specify at runtime:
```python
copilot.load_model(model_name="custom-model")
```

## Testing

```bash
# Test basic functionality
python scripts/copilot.py ask "Hello"

# Test context loading
python scripts/copilot.py explain-portfolio

# Test signal analysis
python scripts/copilot.py analyze-signal AAPL

# Test interactive mode
python scripts/copilot.py chat
```

## Troubleshooting

**Model won't load:**
- Check internet connection (first download)
- Verify sufficient RAM/VRAM
- Try smaller model or CPU mode

**Slow responses:**
- Pre-load model with `load_model`
- Use GPU if available
- Try smaller model (3B vs 7B)

**Out of memory:**
- Use CPU mode: `--cpu`
- Use 4-bit quantization (automatic)
- Try smaller model

## Integration Points

The copilot integrates with:
- `src/database/` - Position and trade data
- `src/signals/` - Signal history
- `src/models/` - ML models for SHAP analysis
- `src/core/config.py` - Configuration access
- MLflow - Model registry (planned)

## Future Enhancements

- [ ] Voice interface (speech-to-text)
- [ ] Multi-modal analysis (chart images)
- [ ] Automated strategy discovery
- [ ] News sentiment integration
- [ ] Economic calendar awareness
- [ ] Risk scenario simulation
- [ ] Backtesting recommendations

## Documentation

Full documentation: `docs/COPILOT.md`

## License

Same as QuantCLI main project.
