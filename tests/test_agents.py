"""Tests for agent implementations."""

import pytest
from socialsim.agents.base import BaseAgent, RandomAgent, SimpleReflexAgent
from socialsim.core.types import AgentProfile, LLMConfig
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

@pytest.fixture
def sample_profile():
    """Create sample agent profile for testing."""
    return AgentProfile(
        agent_id="test_agent",
        name="Test Agent",
        age=30,
        occupation="tester"
    )


@pytest.fixture
def llm_config():
    """Create LLM config for testing."""
    return {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "temperature": 0.7
    }


class TestBaseAgent:
    """Tests for BaseAgent class."""
    
    def test_agent_initialization(self, sample_profile, llm_config):
        """Test agent can be initialized."""
        agent = RandomAgent(sample_profile, llm_config)
        assert agent.profile.agent_id == "test_agent"
        assert agent.state.energy == 1.0
        assert len(agent.memory) == 0
    
    def test_agent_stats(self, sample_profile, llm_config):
        """Test agent statistics tracking."""
        agent = RandomAgent(sample_profile, llm_config)
        stats = agent.get_stats()
        assert stats["steps_taken"] == 0
        assert stats["actions_taken"] == 0
    
    def test_memory_storage(self, sample_profile, llm_config):
        """Test memory storage."""
        agent = RandomAgent(sample_profile, llm_config)
        agent._store_memory({"test": "data"})
        assert len(agent.memory) == 1
        assert agent.memory[0]["test"] == "data"
    
    def test_memory_limit(self, sample_profile, llm_config):
        """Test memory size limit."""
        agent = RandomAgent(sample_profile, llm_config)
        agent.memory_config = {"max_size": 5}
        
        # Add more than max
        for i in range(10):
            agent._store_memory({"index": i})
        
        assert len(agent.memory) == 5
        assert agent.memory[0]["index"] == 5  # Oldest removed


class TestRandomAgent:
    """Tests for RandomAgent."""
    
    def test_random_agent_step(self, sample_profile, llm_config):
        """Test random agent can take a step."""
        agent = RandomAgent(sample_profile, llm_config)
        
        env_state = {
            "locations": ["home", "work", "park"],
            "time": datetime.now(),
            "step": 0
        }
        
        actions = agent.step(env_state)
        assert isinstance(actions, list)
        assert len(actions) > 0
        assert agent.stats["steps_taken"] == 1


class TestSimpleReflexAgent:
    """Tests for SimpleReflexAgent."""
    
    def test_reflex_agent_initialization(self, sample_profile, llm_config):
        """Test reflex agent has needs."""
        agent = SimpleReflexAgent(sample_profile, llm_config)
        assert "hunger" in agent.state.needs
        assert "energy" in agent.state.needs
        assert "social" in agent.state.needs
    
    def test_need_decay(self, sample_profile, llm_config):
        """Test needs decay over time."""
        agent = SimpleReflexAgent(sample_profile, llm_config)
        initial_hunger = agent.state.needs["hunger"]
        
        agent.perceive({"locations": [], "nearby_agents": []})
        
        assert agent.state.needs["hunger"] < initial_hunger
    
    def test_decision_based_on_needs(self, sample_profile, llm_config):
        """Test agent makes decisions based on needs."""
        agent = SimpleReflexAgent(sample_profile, llm_config)
        
        # Make agent very hungry
        agent.state.needs["hunger"] = 0.1
        agent.state.needs["energy"] = 1.0
        agent.state.needs["social"] = 1.0
        
        perception = agent.perceive({
            "locations": ["home", "store"],
            "nearby_agents": []
        })
        
        decision = agent.decide(perception)
        assert decision["action"] == "eat"