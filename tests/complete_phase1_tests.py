"""
Comprehensive tests for Phase 1 complete implementation (Steps 4-8).
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from socialsim.agents.behaviors.needs import NeedDrivenAgent, AgentDecision
from socialsim.environment.simple import SimpleEnvironment, LocationInfo
from socialsim.core.simulation import Simulation
from socialsim.core.types import AgentProfile, SimulationConfig
from socialsim.tools.metrics import MetricsCollector, StepMetrics

from pathlib import Path

import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_profile():
    """Create sample agent profile."""
    return AgentProfile(
        agent_id="test_agent",
        name="Test Agent",
        age=30,
        occupation="tester",
        personality_traits={
            "openness": 0.7,
            "extraversion": 0.6
        }
    )


@pytest.fixture
def llm_config():
    """Create LLM config."""
    return {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "temperature": 0.7
    }


@pytest.fixture
def simple_environment():
    """Create simple environment with basic locations."""
    env = SimpleEnvironment()
    env.add_location("home", "Home", "residential")
    env.add_location("work", "Office", "workplace")
    env.add_location("park", "Park", "recreation")
    return env


# ============================================================================
# Test NeedDrivenAgent (Step 4)
# ============================================================================

class TestNeedDrivenAgent:
    """Tests for NeedDrivenAgent with LLM integration."""
    
    def test_agent_initialization(self, sample_profile, llm_config):
        """Test agent initializes with needs."""
        agent = NeedDrivenAgent(sample_profile, llm_config)
        
        assert "physiological" in agent.state.needs
        assert "safety" in agent.state.needs
        assert "belonging" in agent.state.needs
        assert "esteem" in agent.state.needs
        assert "self_actualization" in agent.state.needs
        
        # All needs start at 1.0 (satisfied)
        for need, value in agent.state.needs.items():
            assert value == 1.0
    
    def test_need_decay(self, sample_profile, llm_config):
        """Test needs decay over time."""
        agent = NeedDrivenAgent(sample_profile, llm_config)
        
        initial_needs = agent.state.needs.copy()
        
        # Perceive environment (causes decay)
        agent.perceive({"locations": [], "nearby_agents": []})
        
        # All needs should have decayed
        for need, initial_value in initial_needs.items():
            assert agent.state.needs[need] < initial_value
    
    def test_need_hierarchy(self, sample_profile, llm_config):
        """Test that lower-level needs are prioritized."""
        agent = NeedDrivenAgent(sample_profile, llm_config)
        
        # Set physiological very low, others high
        agent.state.needs = {
            "physiological": 0.1,
            "safety": 0.9,
            "belonging": 0.9,
            "esteem": 0.9,
            "self_actualization": 0.9
        }
        
        perception = agent.perceive({
            "locations": ["home", "store", "park"],
            "nearby_agents": []
        })
        
        # Decision should address physiological need
        decision = agent.decide(perception)
        
        assert decision["most_pressing_need"] == "physiological"
    
    def test_rule_based_fallback(self, sample_profile, llm_config):
        """Test rule-based decision making when LLM not used."""
        agent = NeedDrivenAgent(sample_profile, llm_config)
        
        # Set critical need
        agent.state.needs["physiological"] = 0.1
        
        perception = agent.perceive({
            "locations": ["home", "store"],
            "nearby_agents": [],
            "time": datetime.now(),
            "step": 0
        })
        
        # Force rule-based decision
        decision = agent._rule_based_decide(agent.state.needs, perception)
        
        assert decision["action"] in ["eat", "rest"]
        assert decision["reasoning"] != ""
    
    def test_action_updates_needs(self, sample_profile, llm_config):
        """Test that actions update need levels."""
        agent = NeedDrivenAgent(sample_profile, llm_config)
        
        agent.state.needs["physiological"] = 0.5
        
        decision = {"action": "eat", "target": "store"}
        agent.act(decision)
        
        # Eating should increase physiological need
        assert agent.state.needs["physiological"] > 0.5
    
    def test_cost_tracking(self, sample_profile, llm_config):
        """Test LLM cost tracking."""
        agent = NeedDrivenAgent(sample_profile, llm_config)
        
        initial_cost = agent.llm_cost_tracker["estimated_cost_usd"]
        
        # Simulate LLM call
        agent.llm_cost_tracker["total_calls"] += 1
        agent.llm_cost_tracker["total_tokens"] += 500
        agent.llm_cost_tracker["estimated_cost_usd"] += 0.0004
        
        cost_summary = agent.get_cost_summary()
        
        assert cost_summary["total_calls"] == 1
        assert cost_summary["estimated_cost_usd"] > initial_cost


# ============================================================================
# Test SimpleEnvironment (Step 5)
# ============================================================================

class TestSimpleEnvironment:
    """Tests for SimpleEnvironment."""
    
    def test_environment_initialization(self):
        """Test environment initializes correctly."""
        env = SimpleEnvironment()
        
        assert len(env.locations) == 0
        assert env.current_step == 0
        assert isinstance(env.current_time, datetime)
    
    def test_add_location(self, simple_environment):
        """Test adding locations."""
        assert "home" in simple_environment.locations
        assert "work" in simple_environment.locations
        
        loc_info = simple_environment.get_location_info("home")
        assert loc_info.name == "Home"
        assert loc_info.location_type == "residential"
    
    def test_duplicate_location_error(self, simple_environment):
        """Test adding duplicate location raises error."""
        with pytest.raises(ValueError):
            simple_environment.add_location("home", "Home2", "residential")
    
    def test_register_agent(self, simple_environment):
        """Test agent registration."""
        simple_environment.register_agent("agent_001", "home")
        
        assert "agent_001" in simple_environment.agent_locations
        assert simple_environment.get_agent_location("agent_001") == "home"
    
    def test_move_agent(self, simple_environment):
        """Test agent movement."""
        simple_environment.register_agent("agent_001", "home")
        
        success = simple_environment.move_agent("agent_001", "work")
        
        assert success
        assert simple_environment.get_agent_location("agent_001") == "work"
        assert "agent_001" in simple_environment.get_agents_at("work")
        assert "agent_001" not in simple_environment.get_agents_at("home")
    
    def test_capacity_limit(self, simple_environment):
        """Test location capacity limits."""
        # Add location with small capacity
        simple_environment.add_location("cafe", "Cafe", "commercial", capacity=2)
        
        # Add 2 agents
        simple_environment.register_agent("agent_001", "cafe")
        simple_environment.register_agent("agent_002", "cafe")
        
        # Try to add 3rd agent (should fail)
        simple_environment.register_agent("agent_003", "home")
        success = simple_environment.move_agent("agent_003", "cafe")
        
        assert not success  # Should fail due to capacity
    
    def test_get_nearby_agents(self, simple_environment):
        """Test getting nearby agents."""
        simple_environment.register_agent("agent_001", "home")
        simple_environment.register_agent("agent_002", "home")
        simple_environment.register_agent("agent_003", "work")
        
        nearby = simple_environment.get_nearby_agents("agent_001")
        
        assert "agent_002" in nearby
        assert "agent_003" not in nearby
        assert "agent_001" not in nearby  # Shouldn't include self
    
    def test_get_locations_by_type(self, simple_environment):
        """Test filtering locations by type."""
        residential = simple_environment.get_locations_by_type("residential")
        workplace = simple_environment.get_locations_by_type("workplace")
        
        assert "home" in residential
        assert "work" in workplace
        assert len(residential) >= 1
    
    def test_environment_update(self, simple_environment):
        """Test environment time progression."""
        initial_time = simple_environment.current_time
        initial_step = simple_environment.current_step
        
        simple_environment.update()
        
        assert simple_environment.current_step == initial_step + 1
        assert simple_environment.current_time > initial_time
    
    def test_environment_reset(self, simple_environment):
        """Test environment reset."""
        simple_environment.register_agent("agent_001", "home")
        simple_environment.update()
        simple_environment.update()
        
        simple_environment.reset()
        
        assert simple_environment.current_step == 0
        assert len(simple_environment.agent_locations) == 0
        assert len(simple_environment.locations) > 0  # Locations remain


# ============================================================================
# Test Simulation (Step 6)
# ============================================================================

class TestSimulation:
    """Tests for main Simulation class."""
    
    def test_simulation_initialization(self):
        """Test simulation initializes correctly."""
        sim = Simulation("test_sim", {"max_steps": 100})
        
        assert sim.name == "test_sim"
        assert len(sim.agents) == 0
        assert sim.current_step == 0
    
    def test_add_agent(self, sample_profile, llm_config):
        """Test adding agent to simulation."""
        sim = Simulation("test_sim")
        agent = NeedDrivenAgent(sample_profile, llm_config)
        
        sim.add_agent(agent)
        
        assert sample_profile.agent_id in sim.agents
        assert sample_profile.agent_id in sim.environment.agent_locations
    
    def test_add_agents_batch(self, llm_config):
        """Test adding multiple agents."""
        sim = Simulation("test_sim")
        
        agents = []
        for i in range(10):
            profile = AgentProfile(
                agent_id=f"agent_{i}",
                name=f"Agent {i}",
                age=30,
                occupation="test"
            )
            agents.append(NeedDrivenAgent(profile, llm_config))
        
        sim.add_agents_batch(agents)
        
        assert len(sim.agents) == 10
    
    def test_duplicate_agent_error(self, sample_profile, llm_config):
        """Test adding duplicate agent raises error."""
        sim = Simulation("test_sim")
        agent = NeedDrivenAgent(sample_profile, llm_config)
        
        sim.add_agent(agent)
        
        with pytest.raises(ValueError):
            sim.add_agent(agent)
    
    def test_run_simulation(self, llm_config):
        """Test running simulation for multiple steps."""
        sim = Simulation("test_sim", {"max_steps": 10})
        
        # Add simple agent
        profile = AgentProfile(
            agent_id="agent_001",
            name="Test",
            age=30,
            occupation="test"
        )
        agent = NeedDrivenAgent(profile, llm_config)
        sim.add_agent(agent)
        
        # Add location
        sim.environment.add_location("test_loc", "Test", "test")
        
        # Run simulation
        results = sim.run(num_steps=5)
        
        assert sim.current_step == 5
        assert results["steps_completed"] == 5
        assert results["num_agents"] == 1
    
    def test_action_processing(self, sample_profile, llm_config):
        """Test action processing."""
        sim = Simulation("test_sim")
        agent = NeedDrivenAgent(sample_profile, llm_config)
        sim.add_agent(agent)
        
        # Add locations
        sim.environment.add_location("start", "Start", "test")
        sim.environment.add_location("end", "End", "test")
        
        # Move agent to start
        sim.environment.move_agent(sample_profile.agent_id, "start")
        
        # Process move action
        sim._execute_action(sample_profile.agent_id, "move:end")
        
        assert sim.environment.get_agent_location(sample_profile.agent_id) == "end"
    
    def test_get_agents_by_location(self, llm_config):
        """Test getting agents at a location."""
        sim = Simulation("test_sim")
        
        # Add agents
        for i in range(3):
            profile = AgentProfile(
                agent_id=f"agent_{i}",
                name=f"Agent {i}",
                age=30,
                occupation="test"
            )
            agent = NeedDrivenAgent(profile, llm_config)
            sim.add_agent(agent, "home")
        
        agents_at_home = sim.get_agents_by_location("home")
        
        assert len(agents_at_home) == 3
    
    def test_pause_resume(self, sample_profile, llm_config):
        """Test pause and resume functionality."""
        sim = Simulation("test_sim")
        
        sim.pause()
        assert sim.is_paused
        
        sim.resume()
        assert not sim.is_paused
    
    def test_reset(self, sample_profile, llm_config):
        """Test simulation reset."""
        sim = Simulation("test_sim")
        agent = NeedDrivenAgent(sample_profile, llm_config)
        sim.add_agent(agent)
        
        # Run some steps
        sim.run(num_steps=5)
        
        assert sim.current_step == 5
        
        # Reset
        sim.reset()
        
        assert sim.current_step == 0
        assert len(sim.agents) == 1  # Agents remain
        assert agent.stats["steps_taken"] == 0  # Agent reset


# ============================================================================
# Test MetricsCollector (Step 7)
# ============================================================================

class TestMetricsCollector:
    """Tests for MetricsCollector."""
    
    def test_collector_initialization(self):
        """Test metrics collector initializes."""
        collector = MetricsCollector("test_sim")
        
        assert collector.simulation_name == "test_sim"
        assert len(collector.step_metrics) == 0
    
    def test_record_step(self):
        """Test recording step metrics."""
        collector = MetricsCollector("test_sim")
        
        metrics = StepMetrics(
            step=1,
            num_agents=10,
            actions_taken=25,
            llm_calls=8,
            step_duration_seconds=1.5
        )
        
        collector.record_step(metrics)
        
        assert len(collector.step_metrics) == 1
        assert collector.step_metrics[0].step == 1
    
    def test_record_custom_metric(self):
        """Test recording custom metrics."""
        collector = MetricsCollector("test_sim")
        
        collector.record_custom("custom_value", 42)
        collector.record_custom("custom_value", 43)
        
        assert "custom_value" in collector.custom_metrics
        assert len(collector.custom_metrics["custom_value"]) == 2
    
    def test_get_summary(self):
        """Test getting summary statistics."""
        collector = MetricsCollector("test_sim")
        
        # Add some metrics
        for i in range(5):
            metrics = StepMetrics(
                step=i,
                num_agents=10,
                actions_taken=20 + i,
                llm_calls=5,
                step_duration_seconds=1.0
            )
            collector.record_step(metrics)
        
        summary = collector.get_summary()
        
        assert summary["total_steps"] == 5
        assert summary["total_actions"] == 110  # 20+21+22+23+24
        assert summary["average_actions_per_step"] == 22.0
    
    def test_save_csv(self, tmp_path):
        """Test saving metrics to CSV."""
        collector = MetricsCollector("test_sim")
        
        # Add metrics
        metrics = StepMetrics(
            step=1,
            num_agents=10,
            actions_taken=25,
            llm_calls=8
        )
        collector.record_step(metrics)
        
        # Save
        filepath = collector.save_csv()

        print(filepath)
        
        assert filepath != ""
        assert Path(filepath).exists()
        
        # Check content
        import csv
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 1
            assert rows[0]["step"] == "1"
    
    def test_save_json(self, tmp_path):
        """Test saving metrics to JSON."""
        collector = MetricsCollector("test_sim", output_dir=str(tmp_path))
        
        metrics = StepMetrics(
            step=1,
            num_agents=10,
            actions_taken=25
        )
        collector.record_step(metrics)
        
        filepath = collector.save_json()
        
        assert filepath != ""
        assert Path(filepath).exists()
        
        # Check content
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
            assert data["simulation_name"] == "test_sim"
            assert len(data["steps"]) == 1
    
    def test_get_step_metrics(self):
        """Test retrieving metrics for specific step."""
        collector = MetricsCollector("test_sim")
        
        for i in range(5):
            metrics = StepMetrics(step=i, num_agents=10, actions_taken=20)
            collector.record_step(metrics)
        
        step_3 = collector.get_step_metrics(3)
        
        assert step_3 is not None
        assert step_3.step == 3
    
    def test_get_metrics_range(self):
        """Test retrieving metrics for a range."""
        collector = MetricsCollector("test_sim")
        
        for i in range(10):
            metrics = StepMetrics(step=i, num_agents=10, actions_taken=20)
            collector.record_step(metrics)
        
        range_metrics = collector.get_metrics_range(2, 5)
        
        assert len(range_metrics) == 4  # Steps 2, 3, 4, 5
        assert range_metrics[0].step == 2
        assert range_metrics[-1].step == 5