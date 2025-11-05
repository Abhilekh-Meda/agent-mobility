"""
Redis-based persistent LLM cache for Phase 2.

Provides distributed, persistent caching of LLM responses across simulation runs.
"""

import redis
import hashlib
import json
import pickle
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from loguru import logger


class RedisLLMCache:
    """Persistent LLM response cache using Redis.
    
    Features:
    - Persistent across simulation runs
    - Shared across multiple processes
    - TTL-based expiration
    - Compression support
    - Statistics tracking
    """
    
    def __init__(
        self,
        host: str = 'localhost',
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        ttl_seconds: int = 86400 * 7,  # 7 days
        prefix: str = 'socialsim:llm:',
        use_compression: bool = True
    ):
        """Initialize Redis cache.
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password (if required)
            ttl_seconds: Time-to-live for cached items
            prefix: Key prefix for namespacing
            use_compression: Whether to compress cached data
        """
        self.ttl_seconds = ttl_seconds
        self.prefix = prefix
        self.use_compression = use_compression
        
        # Connect to Redis
        try:
            self.redis = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=False  # Handle bytes for compression
            )
            
            # Test connection
            self.redis.ping()
            logger.info(f"Connected to Redis at {host}:{port}")
            
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'errors': 0,
            'bytes_saved': 0
        }
    
    def get(
        self,
        prompt: str,
        model: str,
        temperature: float
    ) -> Optional[str]:
        """Get cached response.
        
        Args:
            prompt: Input prompt
            model: Model name
            temperature: Temperature setting
            
        Returns:
            Cached response or None
        """
        key = self._make_key(prompt, model, temperature)
        
        try:
            cached_data = self.redis.get(key)
            
            if cached_data is None:
                self.stats['misses'] += 1
                return None
            
            # Decompress if needed
            if self.use_compression:
                data = pickle.loads(cached_data)
            else:
                data = json.loads(cached_data.decode('utf-8'))
            
            self.stats['hits'] += 1
            logger.debug(f"Cache hit for key: {key[:16]}...")
            
            return data['response']
            
        except Exception as e:
            logger.error(f"Error getting from cache: {e}")
            self.stats['errors'] += 1
            return None
    
    def set(
        self,
        prompt: str,
        model: str,
        temperature: float,
        response: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Set cached response.
        
        Args:
            prompt: Input prompt
            model: Model name
            temperature: Temperature setting
            response: LLM response to cache
            metadata: Optional metadata
            
        Returns:
            Success status
        """
        key = self._make_key(prompt, model, temperature)
        
        try:
            data = {
                'response': response,
                'model': model,
                'temperature': temperature,
                'cached_at': datetime.now().isoformat(),
                'metadata': metadata or {}
            }
            
            # Compress if enabled
            if self.use_compression:
                cached_data = pickle.dumps(data)
            else:
                cached_data = json.dumps(data).encode('utf-8')
            
            # Store with TTL
            self.redis.setex(key, self.ttl_seconds, cached_data)
            
            self.stats['sets'] += 1
            self.stats['bytes_saved'] += len(cached_data)
            
            logger.debug(f"Cached response ({len(cached_data)} bytes)")
            return True
            
        except Exception as e:
            logger.error(f"Error setting cache: {e}")
            self.stats['errors'] += 1
            return False
    
    def delete(self, prompt: str, model: str, temperature: float) -> bool:
        """Delete cached response.
        
        Args:
            prompt: Input prompt
            model: Model name
            temperature: Temperature setting
            
        Returns:
            Success status
        """
        key = self._make_key(prompt, model, temperature)
        
        try:
            self.redis.delete(key)
            return True
        except Exception as e:
            logger.error(f"Error deleting from cache: {e}")
            return False
    
    def clear_all(self) -> int:
        """Clear all cached items with prefix.
        
        Returns:
            Number of keys deleted
        """
        try:
            pattern = f"{self.prefix}*"
            keys = self.redis.keys(pattern)
            
            if keys:
                deleted = self.redis.delete(*keys)
                logger.info(f"Cleared {deleted} cached items")
                return deleted
            
            return 0
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Statistics dictionary
        """
        hit_rate = (
            self.stats['hits'] / max(1, self.stats['hits'] + self.stats['misses'])
        )
        
        return {
            **self.stats,
            'hit_rate': hit_rate,
            'total_requests': self.stats['hits'] + self.stats['misses'],
            'avg_bytes_per_item': (
                self.stats['bytes_saved'] / max(1, self.stats['sets'])
            )
        }
    
    def get_info(self) -> Dict[str, Any]:
        """Get Redis cache information.
        
        Returns:
            Cache info including size and memory usage
        """
        try:
            # Get all keys with prefix
            pattern = f"{self.prefix}*"
            keys = self.redis.keys(pattern)
            
            # Get memory usage
            info = self.redis.info('memory')
            
            return {
                'total_keys': len(keys),
                'memory_used_bytes': info.get('used_memory', 0),
                'memory_used_mb': info.get('used_memory', 0) / 1024 / 1024,
                'connected': True
            }
            
        except Exception as e:
            logger.error(f"Error getting cache info: {e}")
            return {'connected': False, 'error': str(e)}
    
    def _make_key(self, prompt: str, model: str, temperature: float) -> str:
        """Create cache key from parameters.
        
        Args:
            prompt: Input prompt
            model: Model name
            temperature: Temperature setting
            
        Returns:
            Cache key
        """
        # Create deterministic key
        key_data = f"{model}:{temperature}:{prompt}"
        hash_value = hashlib.sha256(key_data.encode('utf-8')).hexdigest()
        
        return f"{self.prefix}{hash_value}"
    
    def exists(self, prompt: str, model: str, temperature: float) -> bool:
        """Check if key exists in cache.
        
        Args:
            prompt: Input prompt
            model: Model name
            temperature: Temperature setting
            
        Returns:
            True if cached
        """
        key = self._make_key(prompt, model, temperature)
        return bool(self.redis.exists(key))
    
    def set_ttl(self, prompt: str, model: str, temperature: float, ttl: int) -> bool:
        """Update TTL for cached item.
        
        Args:
            prompt: Input prompt
            model: Model name
            temperature: Temperature setting
            ttl: New TTL in seconds
            
        Returns:
            Success status
        """
        key = self._make_key(prompt, model, temperature)
        
        try:
            return bool(self.redis.expire(key, ttl))
        except Exception as e:
            logger.error(f"Error setting TTL: {e}")
            return False
    
    def get_ttl(self, prompt: str, model: str, temperature: float) -> int:
        """Get remaining TTL for cached item.
        
        Args:
            prompt: Input prompt
            model: Model name
            temperature: Temperature setting
            
        Returns:
            Remaining TTL in seconds (-1 if no TTL, -2 if not exists)
        """
        key = self._make_key(prompt, model, temperature)
        
        try:
            return self.redis.ttl(key)
        except Exception as e:
            logger.error(f"Error getting TTL: {e}")
            return -2
    
    def close(self):
        """Close Redis connection."""
        try:
            self.redis.close()
            logger.info("Redis connection closed")
        except Exception as e:
            logger.error(f"Error closing Redis: {e}")
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except:
            pass


class CacheManager:
    """Manager for multiple cache backends.
    
    Supports both in-memory and Redis caching with fallback.
    """
    
    def __init__(
        self,
        use_redis: bool = True,
        redis_config: Optional[Dict[str, Any]] = None,
        memory_cache_size: int = 1000
    ):
        """Initialize cache manager.
        
        Args:
            use_redis: Whether to use Redis
            redis_config: Redis configuration
            memory_cache_size: Size of in-memory fallback cache
        """
        self.use_redis = use_redis
        self.redis_cache: Optional[RedisLLMCache] = None
        self.memory_cache: Dict[str, Any] = {}
        self.memory_cache_size = memory_cache_size
        
        # Initialize Redis if enabled
        if use_redis:
            try:
                config = redis_config or {}
                self.redis_cache = RedisLLMCache(**config)
                logger.info("CacheManager initialized with Redis")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis, using memory cache: {e}")
                self.use_redis = False
        
        if not self.use_redis:
            logger.info("CacheManager initialized with memory cache only")
    
    def get(self, prompt: str, model: str, temperature: float) -> Optional[str]:
        """Get from cache (Redis or memory).
        
        Args:
            prompt: Input prompt
            model: Model name
            temperature: Temperature setting
            
        Returns:
            Cached response or None
        """
        # Try Redis first
        if self.use_redis and self.redis_cache:
            result = self.redis_cache.get(prompt, model, temperature)
            if result is not None:
                return result
        
        # Fallback to memory cache
        key = self._make_memory_key(prompt, model, temperature)
        return self.memory_cache.get(key)
    
    def set(
        self,
        prompt: str,
        model: str,
        temperature: float,
        response: str
    ) -> bool:
        """Set in cache (Redis and memory).
        
        Args:
            prompt: Input prompt
            model: Model name
            temperature: Temperature setting
            response: Response to cache
            
        Returns:
            Success status
        """
        success = True
        
        # Set in Redis
        if self.use_redis and self.redis_cache:
            success = self.redis_cache.set(prompt, model, temperature, response)
        
        # Also set in memory cache
        key = self._make_memory_key(prompt, model, temperature)
        
        # Limit memory cache size
        if len(self.memory_cache) >= self.memory_cache_size:
            # Remove oldest (first) item
            oldest_key = next(iter(self.memory_cache))
            del self.memory_cache[oldest_key]
        
        self.memory_cache[key] = response
        
        return success
    
    def get_stats(self) -> Dict[str, Any]:
        """Get combined cache statistics.
        
        Returns:
            Statistics dictionary
        """
        stats = {
            'using_redis': self.use_redis,
            'memory_cache_size': len(self.memory_cache)
        }
        
        if self.use_redis and self.redis_cache:
            stats['redis'] = self.redis_cache.get_stats()
            stats['redis_info'] = self.redis_cache.get_info()
        
        return stats
    
    def _make_memory_key(self, prompt: str, model: str, temperature: float) -> str:
        """Create key for memory cache."""
        key_data = f"{model}:{temperature}:{prompt}"
        return hashlib.md5(key_data.encode('utf-8')).hexdigest()
    
    def clear_all(self) -> int:
        """Clear all caches.
        
        Returns:
            Number of items cleared
        """
        count = len(self.memory_cache)
        self.memory_cache.clear()
        
        if self.use_redis and self.redis_cache:
            count += self.redis_cache.clear_all()
        
        return count
    
    def close(self):
        """Close connections."""
        if self.redis_cache:
            self.redis_cache.close()