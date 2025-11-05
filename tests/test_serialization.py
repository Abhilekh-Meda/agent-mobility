"""
Tests for Step 11: Agent Serialization & Distribution
"""

import pytest
import json
from datetime import datetime

from socialsim.core.types import AgentProfile, AgentState, Location
from socialsim.agents.base import BaseAgent, RandomAgent, SimpleReflexAgent
from socialsim.agents.behaviors.needs import NeedDrivenAgent


from dotenv import load_dotenv
import os
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
LOCAL_LLM_PATH = os.getenv("LOCAL_LLM_PATH")

# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_profile():
    """Create sample agent profile."""
    return AgentProfile(
        agent_id="test_agent_001",
        name="Test Agent",
        age=30,
        occupation="tester",
        personality_traits={
            "openness": 0.7,
            "extraversion": 0.6,
            "conscientiousness": 0.8
        }
    )


@pytest.fixture
def llm_config():
    """LLM configuration."""
    return {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "temperature": 0.7,
        "max_tokens": 256
    }


# ============================================================================
# Test Base Agent Serialization
# ============================================================================

class TestBaseAgentSerialization:
    """Tests for BaseAgent serialization."""
    
    def test_to_dict_includes_required_fields(self, sample_profile, llm_config):
        """Test that to_dict includes all required fields."""
        agent = RandomAgent(sample_profile, llm_config)
        
        # Add some state
        agent.state.location = Location(latitude=37.7749, longitude=-122.4194)
        agent.state.current_activity = "testing"
        agent.stats["steps_taken"] = 10
        agent.memory.append({"test": "data"})
        
        # Serialize
        data = agent.to_dict()
        
        # Check required fields
        assert 'class_name' in data
        assert 'profile' in data
        assert 'state' in data
        assert 'memory' in data
        assert 'stats' in data
        
        # Verify values
        assert data['class_name'] == 'RandomAgent'
        assert data['profile']['agent_id'] == 'test_agent_001'
        assert data['stats']['steps_taken'] == 10
    
    def test_to_dict_excludes_llm(self, sample_profile, llm_config):
        """Test that LLM object is not included in serialization."""
        agent = RandomAgent(sample_profile, llm_config)
        data = agent.to_dict()
        
        # Should not contain LLM object
        assert 'llm' not in data
        assert hasattr(agent, 'llm')  # But agent still has it
    
    def test_memory_truncation(self, sample_profile, llm_config):
        """Test that memory is truncated to last 100 items."""
        agent = RandomAgent(sample_profile, llm_config)
        
        # Add 150 memories
        for i in range(150):
            agent.memory.append({"index": i})
        
        # Serialize
        data = agent.to_dict()
        
        # Should only have last 100
        assert len(data['memory']) == 100
        assert data['memory'][0]['index'] == 50  # Starts at 50 (items 0-49 dropped)
        assert data['memory'][-1]['index'] == 149
    
    def test_serialized_is_json_compatible(self, sample_profile, llm_config):
        """Test that serialized data is JSON-compatible."""
        agent = RandomAgent(sample_profile, llm_config)
        agent.state.location = Location(latitude=37.7749, longitude=-122.4194)
        
        data = agent.to_dict()
        
        # Should be JSON serializable
        try:
            json_str = json.dumps(data)
            assert len(json_str) > 0
            
            # Should be deserializable
            parsed = json.loads(json_str)
            assert parsed['class_name'] == 'RandomAgent'
        except (TypeError, ValueError) as e:
            pytest.fail(f"Serialization not JSON-compatible: {e}")
    
    def test_from_dict_restores_agent(self, sample_profile, llm_config):
        """Test that from_dict correctly restores an agent."""
        # Create original agent
        original = RandomAgent(sample_profile, llm_config)
        original.state.current_activity = "exploring"
        original.state.energy = 0.75
        original.stats["steps_taken"] = 42
        original.memory.append({"event": "test_event"})
        
        # Serialize
        data = original.to_dict()
        
        # Restore
        restored = RandomAgent.from_dict(data, llm_config)
        
        # Verify restoration
        assert restored.profile.agent_id == original.profile.agent_id
        assert restored.state.current_activity == "exploring"
        assert restored.state.energy == 0.75
        assert restored.stats["steps_taken"] == 42
        assert len(restored.memory) == 1
        assert restored.memory[0]["event"] == "test_event"
    
    def test_from_dict_validates_data(self, llm_config):
        """Test that from_dict validates input data."""
        # Missing required field
        invalid_data = {
            'class_name': 'RandomAgent',
            'state': {}
            # Missing 'profile'
        }
        
        with pytest.raises(ValueError, match="Missing required key"):
            RandomAgent.from_dict(invalid_data, llm_config)
    
    def test_serialization_roundtrip(self, sample_profile, llm_config):
        """Test complete serialization-deserialization roundtrip."""
        # Create agent with diverse state
        agent1 = RandomAgent(sample_profile, llm_config)
        agent1.state.location = Location(latitude=40.7128, longitude=-74.0060)
        agent1.state.energy = 0.6
        agent1.state.inventory = ["item1", "item2"]
        agent1.stats["actions_taken"] = 100
        
        # Roundtrip
        data = agent1.to_dict()
        agent2 = RandomAgent.from_dict(data, llm_config)
        
        # Should match
        assert agent2.profile.agent_id == agent1.profile.agent_id
        assert agent2.state.location.latitude == agent1.state.location.latitude
        assert agent2.state.energy == agent1.state.energy
        assert agent2.state.inventory == agent1.state.inventory
        assert agent2.stats["actions_taken"] == agent1.stats["actions_taken"]
    
    def test_get_serialized_size(self, sample_profile, llm_config):
        """Test getting serialized size."""
        agent = RandomAgent(sample_profile, llm_config)
        
        # Add some data
        for i in range(50):
            agent.memory.append({"step": i, "action": "test"})
        
        size = agent.get_serialized_size()
        
        # Should be reasonable size (KB range)
        assert size > 0
        assert size < 100000  # Less than 100KB for basic agent
        
        print(f"\nAgent serialized size: {size} bytes ({size/1024:.2f} KB)")


# ============================================================================
# Test SimpleReflexAgent Serialization
# ============================================================================

class TestSimpleReflexAgentSerialization:
    """Tests for SimpleReflexAgent serialization."""
    
    def test_reflex_agent_serialization(self, sample_profile, llm_config):
        """Test SimpleReflexAgent serialization."""
        agent = SimpleReflexAgent(sample_profile, llm_config)
        
        # Set some state
        agent.state.needs = {
            "hunger": 0.5,
            "energy": 0.7,
            "social": 0.6
        }
        
        # Serialize and restore
        data = agent.to_dict()
        restored = SimpleReflexAgent.from_dict(data, llm_config)
        
        # Check needs are preserved
        assert restored.state.needs["hunger"] == 0.5
        assert restored.state.needs["energy"] == 0.7
        assert restored.state.needs["social"] == 0.6


# ============================================================================
# Test NeedDrivenAgent Serialization
# ============================================================================

class TestNeedDrivenAgentSerialization:
    """Tests for NeedDrivenAgent serialization."""
    
    def test_need_driven_to_dict_includes_extras(self, sample_profile, llm_config):
        """Test that NeedDrivenAgent includes additional fields."""
        agent = NeedDrivenAgent(sample_profile, llm_config)
        
        # Modify some state
        agent.llm_cost_tracker["total_calls"] = 50
        agent.llm_cost_tracker["estimated_cost_usd"] = 0.25
        
        data = agent.to_dict()
        
        # Should include NeedDrivenAgent-specific fields
        assert 'need_decay_rates' in data
        assert 'llm_cost_tracker' in data
        assert data['llm_cost_tracker']['total_calls'] == 50
    
    def test_need_driven_from_dict_restores_extras(self, sample_profile, llm_config):
        """Test that NeedDrivenAgent restores additional fields."""
        # Create and serialize
        agent1 = NeedDrivenAgent(sample_profile, llm_config)
        agent1.llm_cost_tracker["total_calls"] = 75
        agent1.llm_cost_tracker["total_tokens"] = 15000
        agent1.need_decay_rates["physiological"] = 0.02
        
        data = agent1.to_dict()
        
        # Restore
        agent2 = NeedDrivenAgent.from_dict(data, llm_config)
        
        # Verify restoration
        assert agent2.llm_cost_tracker["total_calls"] == 75
        assert agent2.llm_cost_tracker["total_tokens"] == 15000
        assert agent2.need_decay_rates["physiological"] == 0.02
    
    def test_need_driven_serialization_preserves_needs(self, sample_profile, llm_config):
        """Test that need levels are preserved."""
        agent1 = NeedDrivenAgent(sample_profile, llm_config)
        
        # Modify needs
        agent1.state.needs = {
            "physiological": 0.4,
            "safety": 0.8,
            "belonging": 0.6,
            "esteem": 0.7,
            "self_actualization": 0.5
        }
        
        # Roundtrip
        data = agent1.to_dict()
        agent2 = NeedDrivenAgent.from_dict(data, llm_config)
        
        # All needs should match
        for need, value in agent1.state.needs.items():
            assert agent2.state.needs[need] == value


# ============================================================================
# Test Serialization Performance
# ============================================================================

class TestSerializationPerformance:
    """Performance tests for serialization."""
    
    def test_serialization_speed(self, sample_profile, llm_config):
        """Test that serialization is fast."""
        import time
        
        agent = NeedDrivenAgent(sample_profile, llm_config)
        
        # Add some memory
        for i in range(100):
            agent.memory.append({"step": i, "data": "x" * 100})
        
        # Time serialization
        start = time.time()
        for _ in range(100):
            data = agent.to_dict()
        duration = time.time() - start
        
        avg_time_ms = (duration / 100) * 1000
        
        # Should be fast (<10ms per agent)
        assert avg_time_ms < 10, f"Serialization too slow: {avg_time_ms:.2f}ms"
        
        print(f"\nSerialization speed: {avg_time_ms:.2f}ms per agent")
    
    def test_deserialization_speed(self, sample_profile, llm_config):
        """Test that deserialization is fast."""
        import time
        
        agent = NeedDrivenAgent(sample_profile, llm_config)
        data = agent.to_dict()
        
        # Time deserialization
        start = time.time()
        for _ in range(100):
            restored = NeedDrivenAgent.from_dict(data, llm_config)
        duration = time.time() - start
        
        avg_time_ms = (duration / 100) * 1000
        
        # Should be fast (<20ms per agent)
        assert avg_time_ms < 20, f"Deserialization too slow: {avg_time_ms:.2f}ms"
        
        print(f"\nDeserialization speed: {avg_time_ms:.2f}ms per agent")
    
    def test_serialized_size_scaling(self, llm_config):
        """Test that serialized size scales reasonably."""
        sizes = []
        
        for age in [20, 30, 40, 50, 60]:
            profile = AgentProfile(
                agent_id=f"agent_{age}",
                name=f"Agent {age}",
                age=age,
                occupation="test"
            )
            
            agent = NeedDrivenAgent(profile, llm_config)
            
            # Add memory
            for i in range(100):
                agent.memory.append({"step": i})
            
            size = agent.get_serialized_size()
            sizes.append(size)
        
        # Size should be consistent (same memory size)
        avg_size = sum(sizes) / len(sizes)
        for size in sizes:
            assert abs(size - avg_size) < avg_size * 0.1  # Within 10%
        
        print(f"\nAverage serialized size: {avg_size/1024:.2f} KB")
        print(f"Size range: {min(sizes)/1024:.2f} - {max(sizes)/1024:.2f} KB")


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestSerializationEdgeCases:
    """Test edge cases in serialization."""
    
    def test_empty_memory(self, sample_profile, llm_config):
        """Test serialization with empty memory."""
        agent = RandomAgent(sample_profile, llm_config)
        agent.memory = []
        
        data = agent.to_dict()
        restored = RandomAgent.from_dict(data, llm_config)
        
        assert len(restored.memory) == 0
    
    def test_minimal_state(self, llm_config):
        """Test with minimal agent state."""
        profile = AgentProfile(
            agent_id="minimal",
            name="Minimal",
            age=25,
            occupation="test"
        )
        
        agent = RandomAgent(profile, llm_config)
        
        data = agent.to_dict()
        restored = RandomAgent.from_dict(data, llm_config)
        
        assert restored.profile.agent_id == "minimal"
    
    def test_unicode_in_memory(self, sample_profile, llm_config):
        """Test serialization with unicode characters."""
        agent = RandomAgent(sample_profile, llm_config)
        agent.memory.append({"text": "Hello 世界 🌍"})
        
        data = agent.to_dict()
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        
        restored = RandomAgent.from_dict(parsed, llm_config)
        assert restored.memory[0]["text"] == "Hello 世界 🌍"
    
    def test_large_memory(self, sample_profile, llm_config):
        """Test with large memory (should truncate)."""
        agent = RandomAgent(sample_profile, llm_config)
        
        # Add 500 memories
        for i in range(500):
            agent.memory.append({"index": i, "data": "x" * 1000})
        
        data = agent.to_dict()
        
        # Should be truncated to 100
        assert len(data['memory']) == 100
        
        # Should be last 100
        assert data['memory'][0]['index'] == 400


# ============================================================================
# Integration Tests
# ============================================================================

class TestSerializationIntegration:
    """Integration tests for serialization in realistic scenarios."""
    
    def test_serialize_multiple_agent_types(self, llm_config):
        """Test serializing different agent types."""
        agents = []
        
        for i, agent_class in enumerate([RandomAgent, SimpleReflexAgent, NeedDrivenAgent]):
            profile = AgentProfile(
                agent_id=f"agent_{i}",
                name=f"Agent {i}",
                age=30,
                occupation="test"
            )
            agent = agent_class(profile, llm_config)
            agents.append(agent)
        
        # Serialize all
        serialized = [agent.to_dict() for agent in agents]
        
        # Restore all
        restored = [
            RandomAgent.from_dict(serialized[0], llm_config),
            SimpleReflexAgent.from_dict(serialized[1], llm_config),
            NeedDrivenAgent.from_dict(serialized[2], llm_config)
        ]
        
        # Verify
        for original, restored_agent in zip(agents, restored):
            assert restored_agent.profile.agent_id == original.profile.agent_id
    
    def test_serialization_after_simulation_steps(self, sample_profile, llm_config):
        """Test serialization after agent has taken steps."""
        agent = NeedDrivenAgent(sample_profile, llm_config)
        
        # Simulate some steps
        for step in range(10):
            env_state = {
                'step': step,
                'locations': ['home', 'work'],
                'nearby_agents': [],
                'time': datetime.now()
            }
            agent.step(env_state)
        
        # Serialize
        data = agent.to_dict()
        
        # Restore
        restored = NeedDrivenAgent.from_dict(data, llm_config)
        
        # Stats should be preserved
        assert restored.stats['steps_taken'] == 10
        assert restored.stats['actions_taken'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])