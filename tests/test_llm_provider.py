"""Tests for LLM provider integration."""

import pytest
import os
from socialsim.llm.providers import LLMProvider, LLMCache
from dotenv import load_dotenv

load_dotenv()

class TestLLMProvider:
    """Tests for LLM provider factory."""
    
    def test_create_openai_llm(self):
        """Test creating OpenAI LLM."""
        # Skip if no API key
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("No OPENAI_API_KEY in environment")
        
        llm = LLMProvider.create(
            provider="openai",
            model="gpt-4o-mini",
            temperature=0.7
        )
        assert llm is not None
    
    def test_unknown_provider_raises_error(self):
        """Test unknown provider raises ValueError."""
        with pytest.raises(ValueError):
            LLMProvider.create(
                provider="unknown",
                model="some-model"
            )


class TestLLMCache:
    """Tests for LLM response cache."""
    
    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = LLMCache()
        result = cache.get("test prompt")
        assert result is None
        assert cache.misses == 1
    
    def test_cache_hit(self):
        """Test cache hit returns cached value."""
        cache = LLMCache()
        cache.set("test prompt", "test response")
        
        result = cache.get("test prompt")
        assert result == "test response"
        assert cache.hits == 1
    
    def test_cache_size_limit(self):
        """Test cache respects size limit."""
        cache = LLMCache(max_size=3)
        
        cache.set("prompt1", "response1")
        cache.set("prompt2", "response2")
        cache.set("prompt3", "response3")
        cache.set("prompt4", "response4")  # Should evict prompt1
        
        assert len(cache.cache) == 3
        assert cache.get("prompt1") is None  # Evicted
        assert cache.get("prompt4") == "response4"
    
    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        cache = LLMCache()
        
        cache.set("prompt", "response")
        cache.get("prompt")  # Hit
        cache.get("prompt")  # Hit
        cache.get("other")   # Miss
        
        assert cache.hit_rate() == 2/3  # 2 hits, 1 miss