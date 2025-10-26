"""Tests for core type definitions."""

import pytest
from datetime import datetime
from socialsim.core.types import (
    Location,
    AgentProfile,
    AgentState,
    SimulationConfig,
    Action,
    ActionType,
    LLMConfig,
)


class TestLocation:
    """Tests for Location model."""
    
    def test_valid_location(self):
        """Test creating valid location."""
        loc = Location(latitude=37.7749, longitude=-122.4194)
        assert loc.latitude == 37.7749
        assert loc.longitude == -122.4194
    
    def test_invalid_latitude(self):
        """Test invalid latitude raises error."""
        with pytest.raises(ValueError):
            Location(latitude=100.0, longitude=0.0)
        
        with pytest.raises(ValueError):
            Location(latitude=-100.0, longitude=0.0)
    
    def test_invalid_longitude(self):
        """Test invalid longitude raises error."""
        with pytest.raises(ValueError):
            Location(latitude=0.0, longitude=200.0)


class TestAgentProfile:
    """Tests for AgentProfile model."""
    
    def test_valid_profile(self):
        """Test creating valid agent profile."""
        profile = AgentProfile(
            agent_id="agent_001",
            name="Alice",
            age=30,
            occupation="engineer"
        )
        assert profile.agent_id == "agent_001"
        assert profile.name == "Alice"
        assert profile.age == 30
    
    def test_personality_traits_validation(self):
        """Test personality traits are validated."""
        # Valid traits
        profile = AgentProfile(
            agent_id="agent_001",
            name="Alice",
            age=30,
            personality_traits={"openness": 0.7, "extraversion": 0.5}
        )
        assert profile.personality_traits["openness"] == 0.7
        
        # Invalid trait value
        with pytest.raises(ValueError):
            AgentProfile(
                agent_id="agent_001",
                name="Alice",
                age=30,
                personality_traits={"openness": 1.5}  # > 1.0
            )
    
    def test_invalid_age(self):
        """Test invalid age raises error."""
        with pytest.raises(ValueError):
            AgentProfile(
                agent_id="agent_001",
                name="Alice",
                age=-5
            )


class TestAgentState:
    """Tests for AgentState model."""
    
    def test_default_state(self):
        """Test default agent state."""
        state = AgentState()
        assert state.current_activity == "idle"
        assert state.energy == 1.0
        assert len(state.needs) == 0
    
    def test_needs_validation(self):
        """Test needs are validated to 0-1 range."""
        state = AgentState(needs={"hunger": 0.5, "social": 0.8})
        assert state.needs["hunger"] == 0.5
        
        with pytest.raises(ValueError):
            AgentState(needs={"hunger": 1.5})
    
    def test_relationships_validation(self):
        """Test relationships are validated to -1 to 1 range."""
        state = AgentState(relationships={"agent_002": 0.7})
        assert state.relationships["agent_002"] == 0.7
        
        with pytest.raises(ValueError):
            AgentState(relationships={"agent_002": 2.0})


class TestSimulationConfig:
    """Tests for SimulationConfig model."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = SimulationConfig(name="test_sim")
        assert config.name == "test_sim"
        assert config.time_step_seconds == 60
        assert config.max_steps == 1000
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = SimulationConfig(
            name="custom_sim",
            time_step_seconds=300,
            max_steps=500
        )
        assert config.time_step_seconds == 300
        assert config.max_steps == 500


class TestAction:
    """Tests for Action model."""
    
    def test_action_to_string(self):
        """Test converting action to string."""
        action = Action(
            action_type=ActionType.MOVE,
            agent_id="agent_001",
            target="park"
        )
        assert action.to_string() == "move:park"
    
    def test_action_from_string(self):
        """Test parsing action from string."""
        action = Action.from_string("move:park", "agent_001")
        assert action.action_type == ActionType.MOVE
        assert action.target == "park"
        assert action.agent_id == "agent_001"