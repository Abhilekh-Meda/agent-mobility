"""
LLM provider integration for SocialSim.

Provides unified interface for different LLM providers via LangChain.
"""

import os
from typing import Optional
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from loguru import logger


class LLMProvider:
    """Factory for creating LLM instances."""
    
    @staticmethod
    def create(
        provider: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = 512,
        api_key: Optional[str] = None,
        **kwargs
    ) -> BaseChatModel:
        """Create LLM instance from provider specification.
        
        Args:
            provider: Provider name ('openai', 'anthropic', 'gemini', 'local')
            model: Model identifier
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
            api_key: API key (falls back to environment variables)
            **kwargs: Additional provider-specific arguments
            
        Returns:
            Initialized LLM instance
            
        Raises:
            ValueError: If provider is unknown
            RuntimeError: If API key is missing
        """
        provider = provider.lower()
        
        # Get API key from parameter or environment
        if api_key is None:
            api_key = LLMProvider._get_api_key(provider)
        
        if provider == "openai":
            return LLMProvider._create_openai(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
                **kwargs
            )
        elif provider == "anthropic":
            return LLMProvider._create_anthropic(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
                **kwargs
            )
        elif provider == "gemini":
            return LLMProvider._create_gemini(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
                **kwargs
            )
        elif provider == "local":
            return LLMProvider._create_local(
                model=model,
                temperature=temperature,
                **kwargs
            )
        else:
            raise ValueError(
                f"Unknown provider: {provider}. "
                f"Supported providers: openai, anthropic, gemini, local"
            )
    
    @staticmethod
    def _get_api_key(provider: str) -> Optional[str]:
        """Get API key from environment variables.
        
        Args:
            provider: Provider name
            
        Returns:
            API key or None
        """
        env_vars = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "gemini": "GOOGLE_API_KEY"
        }
        
        env_var = env_vars.get(provider)
        if env_var:
            api_key = os.getenv(env_var)
            if not api_key:
                logger.warning(
                    f"No API key found in environment variable {env_var}. "
                    f"Set it or pass api_key parameter."
                )
            return api_key
        return None
    
    @staticmethod
    def _create_openai(
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        api_key: Optional[str],
        **kwargs
    ) -> ChatOpenAI:
        """Create OpenAI LLM instance.
        
        Args:
            model: Model name (e.g., 'gpt-4o-mini', 'gpt-4')
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
            api_key: OpenAI API key
            **kwargs: Additional arguments
            
        Returns:
            ChatOpenAI instance
        """
        logger.info(f"Initializing OpenAI LLM: {model}")
        
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            **kwargs
        )
    
    @staticmethod
    def _create_anthropic(
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        api_key: Optional[str],
        **kwargs
    ) -> BaseChatModel:
        """Create Anthropic LLM instance.
        
        Args:
            model: Model name (e.g., 'claude-3-5-sonnet-20241022')
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
            api_key: Anthropic API key
            **kwargs: Additional arguments
            
        Returns:
            ChatAnthropic instance
        """
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError(
                "langchain-anthropic not installed. "
                "Install with: pip install langchain-anthropic"
            )
        
        logger.info(f"Initializing Anthropic LLM: {model}")
        
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            **kwargs
        )
    
    @staticmethod
    def _create_gemini(
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        api_key: Optional[str],
        **kwargs
    ) -> BaseChatModel:
        """Create Google Gemini LLM instance.
        
        Args:
            model: Model name (e.g., 'gemini-pro', 'gemini-1.5-pro')
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
            api_key: Google API key
            **kwargs: Additional arguments
            
        Returns:
            ChatGoogleGenerativeAI instance
        """
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError(
                "langchain-google-genai not installed. "
                "Install with: pip install langchain-google-genai"
            )
        
        logger.info(f"Initializing Gemini LLM: {model}")
        
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            max_output_tokens=max_tokens,
            google_api_key=api_key,
            **kwargs
        )
    
    @staticmethod
    def _create_local(
        model: str,
        temperature: float,
        **kwargs
    ) -> BaseChatModel:
        """Create local LLM instance (e.g., Ollama).
        
        Args:
            model: Model name (e.g., 'llama2', 'mistral')
            temperature: Sampling temperature
            **kwargs: Additional arguments
            
        Returns:
            Ollama instance
        """
        try:
            from langchain_community.llms import Ollama
        except ImportError:
            raise ImportError(
                "langchain-community not installed. "
                "Install with: pip install langchain-community"
            )
        
        logger.info(f"Initializing local LLM: {model}")
        
        return Ollama(
            model=model,
            temperature=temperature,
            **kwargs
        )

#TODO: get rid of cache. Most likely will do. Need random and varied answers for this to work.
#We are not asking objective questions.
class LLMCache:
    """Simple cache for LLM responses.
    
    Caches responses based on prompt hash to reduce API costs.
    In Phase 1, this is a simple in-memory dict. Later phases
    will add persistent caching (Redis, SQLite).
    """
    
    def __init__(self, max_size: int = 1000):
        """Initialize cache.
        
        Args:
            max_size: Maximum number of cached responses
        """
        self.cache: dict = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def get(self, prompt: str) -> Optional[str]:
        """Get cached response for prompt.
        
        Args:
            prompt: The prompt to look up
            
        Returns:
            Cached response or None
        """
        key = self._hash_prompt(prompt)
        
        if key in self.cache:
            self.hits += 1
            logger.debug(f"Cache hit for prompt (hit rate: {self.hit_rate():.2%})")
            return self.cache[key]
        
        self.misses += 1
        return None
    
    def set(self, prompt: str, response: str) -> None:
        """Cache a response.
        
        Args:
            prompt: The prompt
            response: The LLM response
        """
        key = self._hash_prompt(prompt)
        
        # Simple LRU: remove oldest if full
        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = response
        logger.debug(f"Cached response (cache size: {len(self.cache)})")
    
    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        logger.info("Cache cleared")
    
    def hit_rate(self) -> float:
        """Calculate cache hit rate.
        
        Returns:
            Hit rate as float between 0 and 1
        """
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def stats(self) -> dict:
        """Get cache statistics.
        
        Returns:
            Dictionary of cache stats
        """
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate()
        }
    
    @staticmethod
    def _hash_prompt(prompt: str) -> str:
        """Create hash key for prompt.
        
        Args:
            prompt: The prompt text
            
        Returns:
            Hash string
        """
        import hashlib
        return hashlib.md5(prompt.encode()).hexdigest()