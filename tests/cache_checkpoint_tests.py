"""
Tests for Steps 13-14: Redis Cache and Checkpointing
"""

import pytest
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Check if Redis is available
try:
    import redis
    r = redis.Redis(host='localhost', port=6379)
    r.ping()
    REDIS_AVAILABLE = True
except:
    REDIS_AVAILABLE = False


# ============================================================================
# Test Redis LLM Cache (Step 13)
# ============================================================================

@pytest.mark.skipif(not REDIS_AVAILABLE, reason="Redis not available")
class TestRedisLLMCache:
    """Tests for Redis-based LLM cache."""
    
    def test_redis_cache_initialization(self):
        """Test Redis cache initializes."""
        from socialsim.llm.redis_cache import RedisLLMCache
        
        cache = RedisLLMCache()
        assert cache.redis.ping()
        
        cache.close()
    
    def test_cache_set_and_get(self):
        """Test setting and getting from cache."""
        from socialsim.llm.redis_cache import RedisLLMCache
        
        cache = RedisLLMCache()
        
        # Set
        success = cache.set(
            prompt="Test prompt",
            model="gpt-4o-mini",
            temperature=0.7,
            response="Test response"
        )
        assert success
        
        # Get
        result = cache.get(
            prompt="Test prompt",
            model="gpt-4o-mini",
            temperature=0.7
        )
        
        assert result == "Test response"
        
        cache.close()
    
    def test_cache_miss(self):
        """Test cache miss returns None."""
        from socialsim.llm.redis_cache import RedisLLMCache
        
        cache = RedisLLMCache()
        
        result = cache.get(
            prompt="Non-existent prompt",
            model="gpt-4o-mini",
            temperature=0.7
        )
        
        assert result is None
        cache.close()
    
    def test_cache_statistics(self):
        """Test cache statistics tracking."""
        from socialsim.llm.redis_cache import RedisLLMCache
        
        cache = RedisLLMCache()
        
        # Set some values
        for i in range(5):
            cache.set(f"prompt_{i}", "gpt-4o-mini", 0.7, f"response_{i}")
        
        # Get some (hit)
        for i in range(3):
            cache.get(f"prompt_{i}", "gpt-4o-mini", 0.7)
        
        # Get non-existent (miss)
        cache.get("non_existent", "gpt-4o-mini", 0.7)
        
        stats = cache.get_stats()
        
        assert stats['hits'] == 3
        assert stats['misses'] == 1
        assert stats['sets'] == 5
        assert stats['hit_rate'] == 0.75
        
        cache.close()
    
    def test_cache_ttl(self):
        """Test TTL-based expiration."""
        from socialsim.llm.redis_cache import RedisLLMCache
        
        cache = RedisLLMCache(ttl_seconds=2)
        
        cache.set("test", "gpt-4o-mini", 0.7, "response")
        
        # Should exist immediately
        assert cache.exists("test", "gpt-4o-mini", 0.7)
        
        # Check TTL
        ttl = cache.get_ttl("test", "gpt-4o-mini", 0.7)
        assert 0 < ttl <= 2
        
        # Wait for expiration
        time.sleep(3)
        
        # Should be expired
        result = cache.get("test", "gpt-4o-mini", 0.7)
        assert result is None
        
        cache.close()
    
    def test_cache_with_compression(self):
        """Test cache with compression enabled."""
        from socialsim.llm.redis_cache import RedisLLMCache
        
        cache = RedisLLMCache(use_compression=True)
        
        large_response = "x" * 10000
        
        cache.set("large", "gpt-4o-mini", 0.7, large_response)
        result = cache.get("large", "gpt-4o-mini", 0.7)
        
        assert result == large_response
        
        cache.close()
    
    def test_cache_clear_all(self):
        """Test clearing all cached items."""
        from socialsim.llm.redis_cache import RedisLLMCache
        
        cache = RedisLLMCache(prefix="test:")
        
        # Add some items
        for i in range(10):
            cache.set(f"prompt_{i}", "gpt-4o-mini", 0.7, f"response_{i}")
        
        # Clear
        deleted = cache.clear_all()
        assert deleted == 10
        
        # Verify cleared
        result = cache.get("prompt_0", "gpt-4o-mini", 0.7)
        assert result is None
        
        cache.close()
    
    def test_cache_info(self):
        """Test getting cache information."""
        from socialsim.llm.redis_cache import RedisLLMCache
        
        cache = RedisLLMCache()
        
        info = cache.get_info()
        
        assert 'total_keys' in info
        assert 'memory_used_bytes' in info
        assert info['connected']
        
        cache.close()


@pytest.mark.skipif(not REDIS_AVAILABLE, reason="Redis not available")
class TestCacheManager:
    """Tests for CacheManager."""
    
    def test_cache_manager_with_redis(self):
        """Test cache manager with Redis enabled."""
        from socialsim.llm.redis_cache import CacheManager
        
        manager = CacheManager(use_redis=True)
        
        assert manager.use_redis
        assert manager.redis_cache is not None
        
        manager.close()
    
    def test_cache_manager_fallback(self):
        """Test cache manager fallback to memory."""
        from socialsim.llm.redis_cache import CacheManager
        
        # Use invalid Redis config to force fallback
        manager = CacheManager(
            use_redis=True,
            redis_config={'host': 'invalid-host', 'port': 9999}
        )
        
        # Should fallback to memory
        assert not manager.use_redis
    
    def test_cache_manager_get_set(self):
        """Test cache manager get/set operations."""
        from socialsim.llm.redis_cache import CacheManager
        
        manager = CacheManager(use_redis=True)
        
        manager.set("prompt", "gpt-4o-mini", 0.7, "response")
        result = manager.get("prompt", "gpt-4o-mini", 0.7)
        
        assert result == "response"
        
        manager.close()
    
    def test_cache_manager_stats(self):
        """Test cache manager statistics."""
        from socialsim.llm.redis_cache import CacheManager
        
        manager = CacheManager(use_redis=True)
        
        stats = manager.get_stats()
        
        assert 'using_redis' in stats
        assert 'memory_cache_size' in stats
        
        manager.close()


# ============================================================================
# Test Checkpointing System (Step 14)
# ============================================================================

class TestCheckpointManager:
    """Tests for CheckpointManager."""
    
    @pytest.fixture
    def checkpoint_dir(self, tmp_path):
        """Create temporary checkpoint directory."""
        return tmp_path / "checkpoints"
    
    @pytest.fixture
    def sample_simulation(self):
        """Create sample simulation for testing."""
        from socialsim import Simulation
        from socialsim.agents.behaviors.needs import NeedDrivenAgent
        from socialsim.core.types import AgentProfile
        
        sim = Simulation("test_sim", {"max_steps": 100})
        
        # Add some locations
        sim.environment.add_location("home", "Home", "residential")
        sim.environment.add_location("work", "Work", "workplace")
        
        # Add some agents
        for i in range(5):
            profile = AgentProfile(
                agent_id=f"agent_{i}",
                name=f"Agent {i}",
                age=30,
                occupation="test"
            )
            agent = NeedDrivenAgent(profile, {
                "provider": "openai",
                "model": "gpt-4o-mini"
            })
            sim.add_agent(agent)
        
        # Run a few steps
        sim.run(num_steps=10)
        
        return sim
    
    def test_checkpoint_manager_initialization(self, checkpoint_dir):
        """Test checkpoint manager initializes."""
        from socialsim.core.checkpoint import CheckpointManager
        
        manager = CheckpointManager(checkpoint_dir=str(checkpoint_dir))
        
        assert checkpoint_dir.exists()
    
    def test_save_checkpoint(self, checkpoint_dir, sample_simulation):
        """Test saving checkpoint."""
        from socialsim.core.checkpoint import CheckpointManager
        
        manager = CheckpointManager(checkpoint_dir=str(checkpoint_dir))
        
        filepath = manager.save_checkpoint(sample_simulation)
        
        assert Path(filepath).exists()
        assert Path(filepath).suffix == ".gz"
    
    def test_load_checkpoint(self, checkpoint_dir, sample_simulation):
        """Test loading checkpoint."""
        from socialsim.core.checkpoint import CheckpointManager
        
        manager = CheckpointManager(checkpoint_dir=str(checkpoint_dir))
        
        # Save
        filepath = manager.save_checkpoint(sample_simulation)
        
        # Load
        checkpoint_data = manager.load_checkpoint(filepath)
        
        assert 'metadata' in checkpoint_data
        assert 'agents' in checkpoint_data
        assert 'environment' in checkpoint_data
        assert checkpoint_data['metadata']['num_agents'] == 5
    
    def test_checkpoint_roundtrip(self, checkpoint_dir, sample_simulation):
        """Test save and restore cycle."""
        from socialsim.core.checkpoint import CheckpointManager
        from socialsim import Simulation
        
        manager = CheckpointManager(checkpoint_dir=str(checkpoint_dir))
        
        # Save
        filepath = manager.save_checkpoint(sample_simulation)
        
        # Restore
        restored_sim = manager.restore_simulation(filepath, Simulation)
        
        # Verify
        assert len(restored_sim.agents) == len(sample_simulation.agents)
        assert restored_sim.current_step == sample_simulation.current_step
        assert len(restored_sim.environment.locations) == len(sample_simulation.environment.locations)
    
    def test_checkpoint_metadata(self, checkpoint_dir, sample_simulation):
        """Test checkpoint includes correct metadata."""
        from socialsim.core.checkpoint import CheckpointManager
        
        manager = CheckpointManager(checkpoint_dir=str(checkpoint_dir))
        
        filepath = manager.save_checkpoint(sample_simulation, checkpoint_name="test_checkpoint")
        
        checkpoint_data = manager.load_checkpoint(filepath)
        metadata = checkpoint_data['metadata']
        
        assert metadata['simulation_name'] == "test_sim"
        assert metadata['step'] == 10
        assert metadata['num_agents'] == 5
        assert 'version' in metadata
        assert 'timestamp' in metadata
    
    def test_list_checkpoints(self, checkpoint_dir, sample_simulation):
        """Test listing available checkpoints."""
        from socialsim.core.checkpoint import CheckpointManager
        
        manager = CheckpointManager(checkpoint_dir=str(checkpoint_dir))
        
        # Create multiple checkpoints
        for i in range(3):
            manager.save_checkpoint(sample_simulation, f"checkpoint_{i}")
        
        checkpoints = manager.list_checkpoints()
        
        assert len(checkpoints) == 3
        assert all('name' in cp for cp in checkpoints)
        assert all('size_mb' in cp for cp in checkpoints)
    
    def test_checkpoint_cleanup(self, checkpoint_dir, sample_simulation):
        """Test automatic cleanup of old checkpoints."""
        from socialsim.core.checkpoint import CheckpointManager
        
        manager = CheckpointManager(
            checkpoint_dir=str(checkpoint_dir),
            keep_last_n=3
        )
        
        # Create 5 checkpoints
        for i in range(5):
            manager.save_checkpoint(sample_simulation, f"checkpoint_{i}")
            time.sleep(0.1)  # Ensure different timestamps
        
        # Should only keep last 3
        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) <= 3
    
    def test_delete_checkpoint(self, checkpoint_dir, sample_simulation):
        """Test deleting specific checkpoint."""
        from socialsim.core.checkpoint import CheckpointManager
        
        manager = CheckpointManager(checkpoint_dir=str(checkpoint_dir))
        
        manager.save_checkpoint(sample_simulation, "to_delete")
        
        success = manager.delete_checkpoint("to_delete")
        assert success
        
        checkpoints = manager.list_checkpoints()
        assert not any(cp['name'] == "to_delete" for cp in checkpoints)
    
    def test_checkpoint_compression(self, checkpoint_dir, sample_simulation):
        """Test checkpoint compression."""
        from socialsim.core.checkpoint import CheckpointManager
        
        manager = CheckpointManager(
            checkpoint_dir=str(checkpoint_dir),
            compression_level=9  # Maximum compression
        )
        
        filepath = manager.save_checkpoint(sample_simulation)
        
        file_size = Path(filepath).stat().st_size
        
        # Compressed should be reasonably small
        assert file_size < 10 * 1024 * 1024  # <10MB
        
        print(f"\nCheckpoint size: {file_size / 1024:.2f} KB")
    
    def test_checkpoint_with_metrics(self, checkpoint_dir, sample_simulation):
        """Test checkpoint includes metrics."""
        from socialsim.core.checkpoint import CheckpointManager
        
        manager = CheckpointManager(checkpoint_dir=str(checkpoint_dir))
        
        filepath = manager.save_checkpoint(
            sample_simulation,
            include_metrics=True
        )
        
        checkpoint_data = manager.load_checkpoint(filepath)
        
        assert 'metrics' in checkpoint_data
        assert len(checkpoint_data['metrics']['step_metrics']) > 0


# ============================================================================
# Integration Tests
# ============================================================================

class TestCacheCheckpointIntegration:
    """Integration tests for cache and checkpointing."""
    
    @pytest.mark.skipif(not REDIS_AVAILABLE, reason="Redis not available")
    def test_simulation_with_redis_cache(self):
        """Test simulation using Redis cache."""
        from socialsim import Simulation
        from socialsim.agents.behaviors.needs import NeedDrivenAgent
        from socialsim.core.types import AgentProfile
        from socialsim.llm.redis_cache import CacheManager
        
        # Create cache manager
        cache = CacheManager(use_redis=True)
        
        # Create simulation
        sim = Simulation("cache_test", {})
        sim.environment.add_location("home", "Home", "residential")
        
        # Add agent
        profile = AgentProfile(
            agent_id="agent_001",
            name="Test",
            age=30,
            occupation="test"
        )
        
        agent = NeedDrivenAgent(profile, {
            "provider": "openai",
            "model": "gpt-4o-mini"
        })
        sim.add_agent(agent)
        
        # Run with cache
        sim.run(num_steps=5)
        
        # Check cache was used
        stats = cache.get_stats()
        assert stats['using_redis']
        
        cache.close()
    
    def test_checkpoint_after_long_simulation(self, tmp_path):
        """Test checkpointing after longer simulation."""
        from socialsim import Simulation
        from socialsim.agents.behaviors.needs import NeedDrivenAgent
        from socialsim.core.types import AgentProfile
        from socialsim.core.checkpoint import CheckpointManager
        
        # Create simulation
        sim = Simulation("long_test", {})
        
        for i in range(3):
            sim.environment.add_location(f"loc_{i}", f"Location {i}", "test")
        
        # Add agents
        for i in range(10):
            profile = AgentProfile(
                agent_id=f"agent_{i}",
                name=f"Agent {i}",
                age=30,
                occupation="test"
            )
            agent = NeedDrivenAgent(profile, {
                "provider": "openai",
                "model": "gpt-4o-mini"
            })
            sim.add_agent(agent)
        
        # Run
        sim.run(num_steps=50)
        
        # Checkpoint
        manager = CheckpointManager(checkpoint_dir=str(tmp_path))
        filepath = manager.save_checkpoint(sim)
        
        # Restore
        restored = manager.restore_simulation(filepath, Simulation)
        
        assert restored.current_step == 50
        assert len(restored.agents) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])