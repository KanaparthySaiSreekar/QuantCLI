"""
AI Copilot Service using open-source LLMs.

Provides intelligent assistance for trading decisions, signal analysis,
and system optimization using local LLM models.
"""

import json
import hashlib
from typing import Dict, Any, Optional, List
from pathlib import Path
import threading

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
)
from loguru import logger

from ..core.config import ConfigManager
from .prompts import SYSTEM_PROMPT, get_prompt_template


class CopilotService:
    """
    AI Copilot service using open-source LLMs for trading assistance.

    Features:
    - Local LLM inference (no API costs)
    - Context-aware responses
    - Response caching for performance
    - Memory-efficient model loading
    - Trading domain expertise
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern for model efficiency."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the Copilot service."""
        if self._initialized:
            return

        self.config = ConfigManager()
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self._cache = {}
        self._cache_file = Path.home() / ".quantcli" / "copilot_cache.json"
        self._load_cache()
        self._initialized = True

        logger.info("CopilotService initialized (model will load on first use)")

    def _load_cache(self):
        """Load response cache from disk."""
        try:
            if self._cache_file.exists():
                with open(self._cache_file, 'r') as f:
                    self._cache = json.load(f)
                logger.info(f"Loaded {len(self._cache)} cached responses")
        except Exception as e:
            logger.warning(f"Could not load cache: {e}")
            self._cache = {}

    def _save_cache(self):
        """Save response cache to disk."""
        try:
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            # Keep only the 1000 most recent entries
            if len(self._cache) > 1000:
                items = sorted(self._cache.items(), key=lambda x: x[1].get('timestamp', 0))
                self._cache = dict(items[-1000:])

            with open(self._cache_file, 'w') as f:
                json.dump(self._cache, f)
        except Exception as e:
            logger.warning(f"Could not save cache: {e}")

    def _get_cache_key(self, prompt: str, max_length: int) -> str:
        """Generate cache key for a prompt."""
        key_str = f"{prompt}:{max_length}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_model_name(self) -> str:
        """Get the model name from config or use default."""
        # Check if user specified a model in config
        default_model = "microsoft/Phi-3-mini-4k-instruct"

        # Could also use these alternatives:
        # "meta-llama/Llama-3.2-3B-Instruct" - Fast, good quality
        # "mistralai/Mistral-7B-Instruct-v0.2" - Larger, more capable
        # "HuggingFaceH4/zephyr-7b-beta" - Good for helpfulness

        return default_model

    def load_model(self, model_name: Optional[str] = None, force_cpu: bool = False):
        """
        Load the LLM model for inference.

        Args:
            model_name: HuggingFace model name (default: Phi-3-mini)
            force_cpu: Force CPU inference even if GPU available
        """
        if self.model is not None:
            logger.info("Model already loaded")
            return

        model_name = model_name or self._get_model_name()
        logger.info(f"Loading model: {model_name}")

        try:
            # Check device availability
            device = "cpu"
            if not force_cpu and torch.cuda.is_available():
                device = "cuda"
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            elif not force_cpu and torch.backends.mps.is_available():
                device = "mps"
                logger.info("Using Apple Silicon GPU")
            else:
                logger.info("Using CPU inference")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )

            # Configure quantization for memory efficiency if using GPU
            quantization_config = None
            if device == "cuda":
                try:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    )
                    logger.info("Using 4-bit quantization for memory efficiency")
                except Exception as e:
                    logger.warning(f"Could not enable quantization: {e}")

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto" if device != "cpu" else None,
                trust_remote_code=True,
                torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            )

            if device == "cpu":
                self.model = self.model.to(device)

            # Create pipeline for easy inference
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if device == "cuda" else -1,
            )

            logger.info(f"Model loaded successfully on {device}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def generate_response(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        use_cache: bool = True,
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            prompt: The input prompt
            max_length: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            use_cache: Whether to use cached responses

        Returns:
            Generated text response
        """
        # Check cache first
        if use_cache:
            cache_key = self._get_cache_key(prompt, max_length)
            if cache_key in self._cache:
                logger.info("Using cached response")
                return self._cache[cache_key]['response']

        # Ensure model is loaded
        if self.model is None:
            logger.info("Model not loaded, loading now...")
            self.load_model()

        try:
            # Format prompt with system context
            full_prompt = f"{SYSTEM_PROMPT}\n\nUser: {prompt}\n\nAssistant:"

            # Generate response
            outputs = self.pipeline(
                full_prompt,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            # Extract response
            generated_text = outputs[0]['generated_text']
            response = generated_text.split("Assistant:")[-1].strip()

            # Cache the response
            if use_cache:
                cache_key = self._get_cache_key(prompt, max_length)
                self._cache[cache_key] = {
                    'response': response,
                    'timestamp': __import__('time').time(),
                }
                self._save_cache()

            return response

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: Could not generate response. {str(e)}"

    def analyze_signal(
        self,
        signal_data: Dict[str, Any],
        market_context: Dict[str, Any],
        features: Dict[str, float],
    ) -> str:
        """
        Analyze a trading signal and provide insights.

        Args:
            signal_data: Signal information (symbol, direction, strength, etc.)
            market_context: Current market conditions
            features: Relevant feature values

        Returns:
            Analysis and recommendations
        """
        template = get_prompt_template("signal_analysis")
        prompt = template.format(
            signal_data=json.dumps(signal_data, indent=2),
            market_context=json.dumps(market_context, indent=2),
            features=json.dumps(features, indent=2),
        )

        return self.generate_response(prompt, max_length=768)

    def analyze_portfolio(
        self,
        portfolio_data: Dict[str, Any],
        positions: List[Dict[str, Any]],
        performance_metrics: Dict[str, float],
        market_context: Dict[str, Any],
    ) -> str:
        """
        Analyze portfolio and provide insights.

        Args:
            portfolio_data: Portfolio summary
            positions: Current positions
            performance_metrics: Recent performance
            market_context: Market conditions

        Returns:
            Portfolio analysis and recommendations
        """
        template = get_prompt_template("portfolio_analysis")
        prompt = template.format(
            portfolio_data=json.dumps(portfolio_data, indent=2),
            positions=json.dumps(positions, indent=2),
            performance_metrics=json.dumps(performance_metrics, indent=2),
            market_context=json.dumps(market_context, indent=2),
        )

        return self.generate_response(prompt, max_length=768)

    def analyze_model_performance(
        self,
        metrics: Dict[str, float],
        feature_importance: Dict[str, float],
        predictions: List[Dict[str, Any]],
        shap_values: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Analyze ML model performance.

        Args:
            metrics: Model performance metrics
            feature_importance: Feature importance scores
            predictions: Recent predictions
            shap_values: SHAP values if available

        Returns:
            Model performance analysis
        """
        template = get_prompt_template("model_performance")
        prompt = template.format(
            metrics=json.dumps(metrics, indent=2),
            feature_importance=json.dumps(feature_importance, indent=2),
            predictions=json.dumps(predictions, indent=2),
            shap_values=json.dumps(shap_values or {}, indent=2),
        )

        return self.generate_response(prompt, max_length=768)

    def interpret_market(
        self,
        market_data: Dict[str, Any],
        indicators: Dict[str, float],
        news_sentiment: Optional[Dict[str, Any]] = None,
        historical_context: Optional[str] = None,
    ) -> str:
        """
        Interpret current market conditions.

        Args:
            market_data: Current market data
            indicators: Technical indicators
            news_sentiment: News sentiment data
            historical_context: Historical context

        Returns:
            Market interpretation and insights
        """
        template = get_prompt_template("market_interpretation")
        prompt = template.format(
            market_data=json.dumps(market_data, indent=2),
            indicators=json.dumps(indicators, indent=2),
            news_sentiment=json.dumps(news_sentiment or {}, indent=2),
            historical_context=historical_context or "Not provided",
        )

        return self.generate_response(prompt, max_length=768)

    def recommend_strategy(
        self,
        current_strategy: Dict[str, Any],
        portfolio_state: Dict[str, Any],
        market_conditions: Dict[str, Any],
        performance_history: Dict[str, Any],
    ) -> str:
        """
        Recommend strategy adjustments.

        Args:
            current_strategy: Current trading strategy
            portfolio_state: Current portfolio state
            market_conditions: Market conditions
            performance_history: Historical performance

        Returns:
            Strategy recommendations
        """
        template = get_prompt_template("strategy_recommendation")
        prompt = template.format(
            current_strategy=json.dumps(current_strategy, indent=2),
            portfolio_state=json.dumps(portfolio_state, indent=2),
            market_conditions=json.dumps(market_conditions, indent=2),
            performance_history=json.dumps(performance_history, indent=2),
        )

        return self.generate_response(prompt, max_length=768)

    def ask(self, question: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Ask a general question to the copilot.

        Args:
            question: The question to ask
            context: Optional context information

        Returns:
            Answer to the question
        """
        template = get_prompt_template("general_query")
        prompt = template.format(
            question=question,
            context=json.dumps(context or {}, indent=2),
        )

        return self.generate_response(prompt, max_length=512)

    def clear_cache(self):
        """Clear the response cache."""
        self._cache = {}
        self._save_cache()
        logger.info("Cache cleared")

    def unload_model(self):
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            del self.tokenizer
            del self.pipeline
            self.model = None
            self.tokenizer = None
            self.pipeline = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Model unloaded")
