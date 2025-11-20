# QuantCLI AI Copilot

**Your Intelligent Trading Assistant**

QuantCLI Copilot is a built-in AI assistant powered by open-source LLMs that provides intelligent, context-aware assistance for your quantitative trading system.

---

## 🌟 Features

### Core Capabilities

1. **Natural Language Queries**
   - Ask questions about your portfolio, trades, and strategy in plain English
   - Get intelligent, context-aware responses
   - No API costs - runs locally with open-source models

2. **Signal Analysis**
   - Understand why signals were generated
   - Get feature importance explanations
   - Identify key factors driving trading decisions

3. **Portfolio Insights**
   - Real-time portfolio health assessment
   - Risk concentration analysis
   - Rebalancing recommendations

4. **Market Interpretation**
   - Analyze current market conditions
   - Identify trends and patterns
   - Get strategic recommendations

5. **Model Explainability**
   - SHAP-based feature importance
   - Individual prediction explanations
   - Feature interaction detection
   - Ensemble model analysis

6. **Strategy Recommendations**
   - Parameter optimization suggestions
   - Performance improvement ideas
   - Risk management advice

---

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# The copilot will download the AI model on first use
# Default model: microsoft/Phi-3-mini-4k-instruct (3.8B parameters)
```

### Basic Usage

```bash
# Ask a question
python scripts/copilot.py ask "What's my portfolio status?"

# Analyze a trading signal
python scripts/copilot.py analyze-signal AAPL

# Get portfolio insights
python scripts/copilot.py explain-portfolio

# Market analysis
python scripts/copilot.py market-insight TSLA --days 30

# Interactive chat mode
python scripts/copilot.py chat
```

---

## 📖 Detailed Usage

### 1. Ask Command

Ask any question about your trading system:

```bash
# General questions
python scripts/copilot.py ask "How is my strategy performing?"

# Symbol-specific questions
python scripts/copilot.py ask "Should I buy AAPL?" --symbol AAPL

# With full context
python scripts/copilot.py ask "What are my riskiest positions?" --context
```

**Example Questions:**
- "What's my current portfolio value?"
- "Why did I get a buy signal for TSLA?"
- "How is my win rate this month?"
- "What are the top features in my model?"
- "Should I adjust my risk parameters?"

### 2. Signal Analysis

Analyze recent trading signals for deeper insights:

```bash
# Analyze last 24 hours of signals
python scripts/copilot.py analyze-signal AAPL

# Analyze last 48 hours
python scripts/copilot.py analyze-signal AAPL --hours 48
```

**What You Get:**
- Explanation of why the signal was generated
- Key contributing factors
- Risk considerations
- Recommended position size
- Expected holding period

### 3. Portfolio Analysis

Get comprehensive portfolio insights:

```bash
# Analyze last 30 days
python scripts/copilot.py explain-portfolio

# Analyze last 90 days
python scripts/copilot.py explain-portfolio --days 90
```

**Analysis Includes:**
- Portfolio health assessment
- Risk concentration analysis
- Rebalancing recommendations
- Performance breakdown
- Warning flags or concerns

### 4. Market Insights

Understand market conditions:

```bash
# Get market insights for a symbol
python scripts/copilot.py market-insight AAPL

# Analyze longer timeframe
python scripts/copilot.py market-insight AAPL --days 60
```

**Insights Provided:**
- Market regime assessment (bull/bear/sideways)
- Key trends and patterns
- Potential opportunities
- Risk factors to monitor
- Strategic recommendations

### 5. Interactive Chat

Start a conversation with your AI assistant:

```bash
python scripts/copilot.py chat
```

**Chat Features:**
- Continuous context across questions
- Multi-turn conversations
- Portfolio and market awareness
- Special commands:
  - `exit` or `quit` - End session
  - `clear` - Clear screen
  - `help` - Show help

**Example Conversation:**
```
You: What's my current portfolio status?
Copilot: [Provides portfolio summary with metrics]

You: What's my largest position?
Copilot: [Analyzes and explains largest position]

You: Is it too risky?
Copilot: [Assesses risk and provides recommendations]
```

---

## 🎛️ Advanced Features

### Model Management

```bash
# Pre-load the model (faster subsequent responses)
python scripts/copilot.py load-model

# Use a different model
python scripts/copilot.py load-model --model meta-llama/Llama-3.2-3B-Instruct

# Force CPU inference (if you don't have GPU)
python scripts/copilot.py load-model --cpu
```

### Cache Management

The copilot caches responses for faster performance:

```bash
# Clear the cache
python scripts/copilot.py clear-cache

# Check status
python scripts/copilot.py status
```

### Status and Configuration

```bash
# Check copilot status
python scripts/copilot.py status
```

Shows:
- Model loading status
- Cache size
- Database connection
- Device (GPU/CPU)
- Model name

---

## 🤖 Supported Models

The copilot supports any HuggingFace model. Recommended options:

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| **microsoft/Phi-3-mini-4k-instruct** | 3.8B | ⚡⚡⚡ | ⭐⭐⭐ | Default - Fast & accurate |
| **meta-llama/Llama-3.2-3B-Instruct** | 3B | ⚡⚡⚡ | ⭐⭐⭐ | Very fast, great quality |
| **mistralai/Mistral-7B-Instruct-v0.2** | 7B | ⚡⚡ | ⭐⭐⭐⭐ | More capable, slower |
| **HuggingFaceH4/zephyr-7b-beta** | 7B | ⚡⚡ | ⭐⭐⭐⭐ | Helpful & conversational |

**Hardware Requirements:**
- **CPU only:** Phi-3-mini or Llama-3.2-3B (4-8GB RAM)
- **GPU (8GB VRAM):** Any 7B model with 4-bit quantization
- **GPU (16GB+ VRAM):** Larger models without quantization

---

## 🧠 Model Explainability with SHAP

The copilot uses SHAP (SHapley Additive exPlanations) for interpretable ML:

### Python API

```python
from src.copilot.explainer import ModelExplainer
import numpy as np

# Create explainer
explainer = ModelExplainer()
explainer.create_explainer(
    model=your_xgboost_model,
    background_data=X_train,
    model_type="tree"
)

# Explain a prediction
explanation = explainer.explain_prediction(
    X=feature_vector,
    prediction=0.75
)

# Get feature importance
importance = explainer.get_feature_importance(X_test)

# Detect feature interactions
interactions = explainer.detect_interactions(X_test, top_k=10)

# Generate natural language summary
summary = explainer.generate_summary(explanation, prediction=0.75)
print(summary)
```

### What SHAP Provides

1. **Feature Contributions:** How much each feature affected the prediction
2. **Global Importance:** Which features matter most overall
3. **Interactions:** Which features work together
4. **Transparency:** Understand what the model is thinking

---

## 🔧 Integration with Your Code

### Using CopilotService

```python
from src.copilot.service import CopilotService

# Initialize copilot
copilot = CopilotService()

# Load model (optional - will auto-load on first use)
copilot.load_model()

# Ask a question
response = copilot.ask(
    "Why did the model predict a buy signal?",
    context={"symbol": "AAPL", "confidence": 0.85}
)
print(response)

# Analyze a signal
analysis = copilot.analyze_signal(
    signal_data={"symbol": "AAPL", "direction": "BUY", "strength": 0.8},
    market_context={"price": 150.0, "volume": 1000000},
    features={"rsi": 65, "macd": 0.5}
)
print(analysis)
```

### Using ContextProvider

```python
from src.copilot.context import ContextProvider

# Get trading context
context = ContextProvider()

# Portfolio summary
portfolio = context.get_portfolio_summary()

# Current positions
positions = context.get_current_positions(limit=10)

# Recent trades
trades = context.get_recent_trades(symbol="AAPL", days=7)

# Performance metrics
performance = context.get_performance_metrics(days=30)

# Recent signals
signals = context.get_recent_signals(symbol="AAPL", hours=24)

# Market data
market = context.get_market_data(symbol="AAPL", days=30)

# Full context for copilot
full_context = context.get_full_context(symbol="AAPL")
```

---

## 📊 Example Outputs

### Signal Analysis

```
╭─ Signal Analysis: AAPL ─────────────────────────╮
│                                                  │
│ Analysis of Recent Trading Signals              │
│                                                  │
│ **Signal Summary**                              │
│ - Direction: BUY                                │
│ - Strength: 0.85 (Strong)                       │
│ - Confidence: 78%                               │
│                                                  │
│ **Key Factors**                                 │
│ 1. RSI crossed above 50 (bullish momentum)     │
│ 2. MACD golden cross detected                  │
│ 3. Volume 2x above average (strong conviction) │
│                                                  │
│ **Risk Assessment**                             │
│ - Low risk given strong momentum indicators     │
│ - Stop loss recommended at $145                 │
│                                                  │
│ **Recommendation**                              │
│ Position size: 2-3% of portfolio                │
│ Entry: $150-152 range                           │
│ Target: $165 (8% upside)                        │
│ Expected holding: 2-3 weeks                     │
│                                                  │
╰──────────────────────────────────────────────────╯
```

### Portfolio Analysis

```
Portfolio Summary
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Metric              ┃ Value         ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ Total Value         │ $125,430.50   │
│ Unrealized P&L      │ $5,230.25     │
│ Number of Positions │ 12            │
│ Total Trades        │ 48            │
│ Win Rate            │ 62.5%         │
│ Avg P&L per Trade   │ $108.96       │
└─────────────────────┴───────────────┘

╭─ AI Portfolio Analysis ──────────────────────────╮
│                                                  │
│ **Portfolio Health: GOOD ✓**                   │
│                                                  │
│ Your portfolio shows healthy diversification    │
│ across 12 positions with solid returns (+4.3%). │
│                                                  │
│ **Strengths:**                                  │
│ - Above 60% win rate (industry average: 55%)   │
│ - Good risk/reward with modest drawdowns        │
│ - No over-concentration (largest: 12%)         │
│                                                  │
│ **Recommendations:**                            │
│ 1. Consider taking profits on AAPL (+18%)      │
│ 2. Review TSLA position (down 8%)              │
│ 3. Increase position in tech sector (trending) │
│                                                  │
│ **Risk Assessment:**                            │
│ Overall risk level: MODERATE                    │
│ Portfolio beta: 1.15 (slightly aggressive)      │
│                                                  │
╰──────────────────────────────────────────────────╯
```

---

## 🎯 Best Practices

### 1. Model Selection

- **For quick queries:** Use Phi-3-mini or Llama-3.2-3B
- **For deep analysis:** Use Mistral-7B or Zephyr-7b
- **CPU-only systems:** Stick with 3B models
- **GPU available:** Use 7B models with quantization

### 2. Context Usage

- Use `--context` flag for questions needing full system awareness
- Specify `--symbol` to focus analysis on specific stocks
- For general questions, basic context is sufficient

### 3. Caching

- The copilot caches responses for speed
- Clear cache after major portfolio changes
- Cache persists across sessions

### 4. Performance Optimization

- Pre-load the model if you'll make multiple queries
- Use GPU if available (10-50x faster)
- 4-bit quantization reduces memory by 75%

### 5. Prompt Engineering

Ask specific, focused questions:

✅ **Good:**
- "Why did the RSI indicator trigger a sell signal for AAPL?"
- "What's the risk/reward ratio of my current TSLA position?"
- "Should I increase my position size given the recent breakout?"

❌ **Too Vague:**
- "What should I do?"
- "Is the market good?"
- "Help me trade"

---

## 🔒 Privacy & Security

**All copilot processing happens locally:**
- ✅ No data sent to external APIs
- ✅ No cloud dependencies
- ✅ Your trading data stays private
- ✅ Open-source models and code
- ✅ Fully auditable

**Data Usage:**
- Portfolio data fetched from local database
- Market data from configured providers
- No telemetry or analytics collection

---

## 🐛 Troubleshooting

### Model won't load

```bash
# Check available memory
free -h

# Try CPU mode
python scripts/copilot.py load-model --cpu

# Try smaller model
python scripts/copilot.py load-model --model meta-llama/Llama-3.2-3B-Instruct
```

### Slow responses

```bash
# Pre-load model
python scripts/copilot.py load-model

# Check if using GPU
python scripts/copilot.py status

# Reduce max_length in service.py if needed
```

### Database connection errors

```bash
# Verify database is running
python scripts/init_database.py

# Check config
cat config/database.yaml
```

### Out of memory

```bash
# Use smaller model
python scripts/copilot.py load-model --model microsoft/Phi-3-mini-4k-instruct

# Force CPU (uses less memory)
python scripts/copilot.py load-model --cpu

# Close other applications
```

---

## 🔮 Future Enhancements

Planned features:

1. **Multi-modal Analysis**
   - Chart image understanding
   - Visual portfolio dashboards
   - Technical pattern recognition from images

2. **Advanced Strategies**
   - Automated strategy discovery
   - Hyperparameter optimization suggestions
   - Regime detection and adaptation

3. **Enhanced Context**
   - News sentiment integration
   - Social media trend analysis
   - Economic calendar awareness

4. **Collaborative Features**
   - Share insights with team
   - Backtesting recommendations
   - Risk scenario simulation

5. **Voice Interface**
   - Speech-to-text queries
   - Audio responses
   - Hands-free trading assistant

---

## 📚 Additional Resources

- **SHAP Documentation:** https://shap.readthedocs.io/
- **HuggingFace Models:** https://huggingface.co/models
- **Transformers Library:** https://huggingface.co/docs/transformers/
- **QuantCLI Docs:** See main README.md

---

## 💡 Tips & Tricks

1. **Speed up first query:** Run `load-model` before starting
2. **Best quality:** Use Mistral-7B with GPU
3. **Interactive analysis:** Use `chat` mode for exploratory questions
4. **Explain trades:** Ask "Why?" after any signal or trade
5. **Portfolio reviews:** Run `explain-portfolio` weekly
6. **Market checks:** Use `market-insight` before major trades
7. **Model debugging:** Ask about feature importance and SHAP values

---

## 🤝 Contributing

To improve the copilot:

1. **Add new prompt templates** in `src/copilot/prompts.py`
2. **Enhance context providers** in `src/copilot/context.py`
3. **Improve explainability** in `src/copilot/explainer.py`
4. **Add CLI commands** in `scripts/copilot.py`

All contributions welcome!

---

## 📄 License

Same as QuantCLI main project.

---

**Built with ❤️ for quantitative traders**
