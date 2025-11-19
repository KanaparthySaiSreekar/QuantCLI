"""
Prompt templates for the QuantCLI AI Copilot.

These templates provide domain-specific context for financial trading analysis.
"""

SYSTEM_PROMPT = """You are QuantCLI Copilot, an expert AI assistant for quantitative trading and algorithmic trading systems.

Your expertise includes:
- Technical analysis and trading signals
- Portfolio management and risk assessment
- Machine learning models for financial prediction
- Market microstructure and trading strategies
- Backtesting and performance metrics (Sharpe ratio, Sortino ratio, max drawdown)
- Ensemble models (XGBoost, LightGBM, CatBoost)
- Feature engineering for financial data

Guidelines:
- Provide clear, concise, actionable insights
- Use precise financial terminology
- Back recommendations with reasoning
- Highlight risks and uncertainties
- Be honest when uncertain
- Focus on practical implementation

Context: You are integrated into QuantCLI, an institutional-grade algorithmic trading platform with:
- 7 data sources (Alpha Vantage, Tiingo, FRED, Finnhub, Polygon, Reddit, GDELT)
- 50+ technical indicators
- Ensemble ML models with time-series cross-validation
- MLflow experiment tracking
- PostgreSQL/TimescaleDB for data storage
- Interactive Brokers integration (in development)
"""

SIGNAL_ANALYSIS_PROMPT = """Analyze the following trading signal and provide insights:

Signal Details:
{signal_data}

Market Context:
{market_context}

Recent Features:
{features}

Task: Explain:
1. Why this signal was generated
2. Key factors contributing to the signal strength
3. Risk considerations
4. Recommended position size (if applicable)
5. Expected holding period

Keep the response concise and actionable."""

PORTFOLIO_ANALYSIS_PROMPT = """Analyze the current portfolio and provide insights:

Portfolio Summary:
{portfolio_data}

Current Positions:
{positions}

Recent Performance:
{performance_metrics}

Market Conditions:
{market_context}

Task: Provide:
1. Portfolio health assessment
2. Risk concentration analysis
3. Rebalancing recommendations (if needed)
4. Potential improvements
5. Warning flags or concerns

Be specific and data-driven."""

MODEL_PERFORMANCE_PROMPT = """Analyze the ML model performance and provide insights:

Model Metrics:
{metrics}

Feature Importance:
{feature_importance}

Recent Predictions:
{predictions}

SHAP Values (if available):
{shap_values}

Task: Explain:
1. Model performance quality
2. Key predictive features
3. Potential overfitting or underfitting signals
4. Recommendations for improvement
5. Trust level for current predictions

Focus on actionable insights."""

MARKET_INTERPRETATION_PROMPT = """Interpret the current market conditions:

Market Data:
{market_data}

Technical Indicators:
{indicators}

Recent News/Sentiment:
{news_sentiment}

Historical Context:
{historical_context}

Task: Provide:
1. Current market regime assessment
2. Key trends and patterns
3. Potential opportunities
4. Risk factors to monitor
5. Strategic recommendations

Be specific about timeframes and confidence levels."""

STRATEGY_RECOMMENDATION_PROMPT = """Recommend trading strategies based on current conditions:

Current Strategy:
{current_strategy}

Portfolio State:
{portfolio_state}

Market Conditions:
{market_conditions}

Performance History:
{performance_history}

Task: Suggest:
1. Strategy adjustments (if needed)
2. New opportunities to explore
3. Risk management improvements
4. Parameter tuning recommendations
5. Implementation priorities

Prioritize by impact and feasibility."""

CONFIG_OPTIMIZATION_PROMPT = """Analyze configuration and suggest optimizations:

Current Configuration:
{config}

Performance Metrics:
{metrics}

Resource Usage:
{resources}

Known Issues:
{issues}

Task: Recommend:
1. Parameter optimizations
2. Feature selection improvements
3. Risk parameter adjustments
4. Performance bottleneck fixes
5. Best practices to adopt

Focus on measurable improvements."""

GENERAL_QUERY_PROMPT = """Answer the following question about QuantCLI or trading:

Question: {question}

Available Context:
{context}

Provide a clear, accurate, and helpful response. If you don't have enough information, say so and suggest what information would help."""

EXPLAIN_PREDICTION_PROMPT = """Explain this model prediction:

Prediction: {prediction}
Confidence: {confidence}
Features Used: {features}
SHAP Contributions: {shap_values}

Task: Explain in plain language:
1. What the model predicted and why
2. Which features drove the prediction most
3. How confident should we be
4. What could change this prediction
5. Recommended action based on this prediction

Make it understandable for a quantitative trader."""

ERROR_ANALYSIS_PROMPT = """Analyze this trading system error or issue:

Error/Issue: {error}
Context: {context}
Recent Logs: {logs}
System State: {state}

Task: Provide:
1. Root cause analysis
2. Impact assessment
3. Immediate mitigation steps
4. Long-term fix recommendations
5. Prevention strategies

Be specific and actionable."""

BACKTEST_ANALYSIS_PROMPT = """Analyze these backtesting results:

Backtest Results:
{results}

Performance Metrics:
{metrics}

Drawdown Analysis:
{drawdowns}

Trade Analysis:
{trades}

Task: Assess:
1. Overall strategy quality
2. Risk-adjusted returns analysis
3. Statistical significance of results
4. Potential overfitting indicators
5. Recommendations for improvement

Focus on statistical rigor and practical insights."""


def get_prompt_template(prompt_type: str) -> str:
    """Get the appropriate prompt template based on type."""
    templates = {
        "signal_analysis": SIGNAL_ANALYSIS_PROMPT,
        "portfolio_analysis": PORTFOLIO_ANALYSIS_PROMPT,
        "model_performance": MODEL_PERFORMANCE_PROMPT,
        "market_interpretation": MARKET_INTERPRETATION_PROMPT,
        "strategy_recommendation": STRATEGY_RECOMMENDATION_PROMPT,
        "config_optimization": CONFIG_OPTIMIZATION_PROMPT,
        "general_query": GENERAL_QUERY_PROMPT,
        "explain_prediction": EXPLAIN_PREDICTION_PROMPT,
        "error_analysis": ERROR_ANALYSIS_PROMPT,
        "backtest_analysis": BACKTEST_ANALYSIS_PROMPT,
    }
    return templates.get(prompt_type, GENERAL_QUERY_PROMPT)
