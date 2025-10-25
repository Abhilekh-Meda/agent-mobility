"""
Base agent implementation for SocialSim.

This module defines the abstract BaseAgent class that all agents must inherit from.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from langchain_core.language_models import BaseChatModel
from loguru import logger

from socialsim.core.types import AgentProfile, AgentState, Action, LLMConfig
from socialsim.llm.providers import LLMProvider


class BaseAgent(ABC):
    """Abstract base class for all agents in the simulation.
    
    Agents follow a perceive-decide-act cycle:
    1. Perceive: Filter environment state to agent's perception
    2. Decide: Make decisions based on perception and internal state
    3. Act: Execute actions in the environment
    
    Attributes:
        profile: Static agent attributes (identity)
        state: Dynamic agent state (changes during simulation)
        llm: Language model for decision-making
        memory: List of past experiences (simple for MVP)
    """
    
    def __init__(
        self,
        profile: AgentProfile,
        llm_config: Dict[str, Any],
        memory_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize agent.
        
        Args:
            profile: Agent's static profile
            llm_config: Configuration for LLM provider
            memory_config: Optional memory configuration (unused in Phase 1)
        """
        self.profile = profile
        self.state = AgentState()
        
        # Initialize LLM
        llm_cfg = LLMConfig(**llm_config)
        self.llm = self._initialize_llm(llm_cfg)
        
        # Simple memory for MVP (just a list of events)
        self.memory: List[Dict[str, Any]] = []
        self.memory_config = memory_config or {}
        
        # Statistics
        self.stats = {
            "steps_taken": 0,
            "actions_taken": 0,
            "llm_calls": 0,
            "decisions_made": 0
        }
        
        logger.debug(f"Initialized agent: {self.profile.agent_id}")
    
    def _initialize_llm(self, config: LLMConfig) -> BaseChatModel:
        """Initialize language model from configuration.
        
        Args:
            config: LLM configuration
            
        Returns:
            Initialized LLM instance
        """
        return LLMProvider.create(
            provider=config.provider,
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            api_key=config.api_key
        )
    
    @abstractmethod
    def perceive(self, environment_state: Dict[str, Any]) -> Dict[str, Any]:
        """Filter environment state to agent's perception.
        
        This method should:
        1. Extract relevant information from environment
        2. Update internal state (needs, emotions, etc.)
        3. Return perception dict for decision-making
        
        Args:
            environment_state: Full environment state
            
        Returns:
            Filtered perception dictionary
        """
        pass
    
    @abstractmethod
    def decide(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """Make decision based on perception.
        
        This is where LLM-based reasoning happens. Should return
        a decision dictionary with at least an 'action' key.
        
        Args:
            perception: Output from perceive()
            
        Returns:
            Decision dictionary (must include 'action' key)
        """
        pass
    
    @abstractmethod
    def act(self, decision: Dict[str, Any]) -> List[str]:
        """Convert decision to executable actions.
        
        Transforms the decision dict into a list of action strings
        that the environment can process.
        
        Args:
            decision: Output from decide()
            
        Returns:
            List of action strings in format "action_type:target"
        """
        pass
    
    def step(self, environment_state: Dict[str, Any]) -> List[str]:
        """Execute one complete agent step (perceive -> decide -> act).
        
        This is the main entry point called by the simulation engine.
        
        Args:
            environment_state: Current state of the environment
            
        Returns:
            List of actions to execute
        """
        self.stats["steps_taken"] += 1
        
        try:
            # 1. Perceive environment
            perception = self.perceive(environment_state)
            
            # 2. Make decision
            decision = self.decide(perception)
            self.stats["decisions_made"] += 1
            
            # 3. Execute actions
            actions = self.act(decision)
            self.stats["actions_taken"] += len(actions)
            
            # 4. Store in memory (simple for MVP)
            self._store_memory({
                "step": environment_state.get("step", 0),
                "perception": perception,
                "decision": decision,
                "actions": actions
            })
            
            return actions
            
        except Exception as e:
            logger.error(f"Error in agent {self.profile.agent_id} step: {e}")
            # Return safe fallback action
            return ["rest"]
    
    def _store_memory(self, experience: Dict[str, Any]) -> None:
        """Store experience in memory.
        
        For Phase 1, this is just a simple list. In later phases,
        we'll implement hierarchical memory with compression.
        
        Args:
            experience: Dictionary containing step information
        """
        self.memory.append(experience)
        
        # Simple memory limit for Phase 1
        max_memory = self.memory_config.get("max_size", 100)
        if len(self.memory) > max_memory:
            self.memory = self.memory[-max_memory:]
    
    def get_recent_memory(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get the n most recent memories.
        
        Args:
            n: Number of recent memories to retrieve
            
        Returns:
            List of memory dictionaries
        """
        return self.memory[-n:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics.
        
        Returns:
            Dictionary of agent statistics
        """
        return {
            **self.stats,
            "memory_size": len(self.memory),
            "current_activity": self.state.current_activity,
            "energy": self.state.energy
        }
    
    def reset(self) -> None:
        """Reset agent to initial state.
        
        Useful for running multiple simulations with same agents.
        """
        self.state = AgentState()
        self.memory = []
        self.stats = {
            "steps_taken": 0,
            "actions_taken": 0,
            "llm_calls": 0,
            "decisions_made": 0
        }
        logger.debug(f"Reset agent: {self.profile.agent_id}")
    
    def __str__(self) -> str:
        return (
            f"Agent({self.profile.name}, {self.state.current_activity}, "
            f"energy={self.state.energy:.2f})"
        )
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.profile.agent_id}>"


class RandomAgent(BaseAgent):
    """Simple random agent for testing.
    
    Makes random decisions without using LLM. Useful for:
    - Testing infrastructure
    - Performance benchmarking
    - Debugging
    """
    
    def __init__(self, profile: AgentProfile, llm_config: Dict[str, Any]):
        super().__init__(profile, llm_config)
        self.available_actions = ["move:home", "move:work", "move:park", "rest", "eat"]
    
    def perceive(self, environment_state: Dict[str, Any]) -> Dict[str, Any]:
        """Simple perception - just copy environment state."""
        return {
            "locations": environment_state.get("locations", []),
            "time": environment_state.get("time"),
            "nearby_agents": environment_state.get("nearby_agents", [])
        }
    
    def decide(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """Make random decision."""
        import random
        action = random.choice(self.available_actions)
        return {
            "action": action,
            "reasoning": "random choice"
        }
    
    def act(self, decision: Dict[str, Any]) -> List[str]:
        """Return the chosen action."""
        return [decision["action"]]


class SimpleReflexAgent(BaseAgent):
    """Rule-based reflex agent for testing.
    
    Makes decisions based on simple if-then rules without LLM.
    More realistic than RandomAgent but still deterministic.
    """
    
    def __init__(self, profile: AgentProfile, llm_config: Dict[str, Any]):
        super().__init__(profile, llm_config)
        self.state.needs = {
            "hunger": 1.0,
            "energy": 1.0,
            "social": 1.0
        }
    
    def perceive(self, environment_state: Dict[str, Any]) -> Dict[str, Any]:
        """Perceive environment and update needs."""
        # Decay needs over time
        self.state.needs["hunger"] = max(0, self.state.needs["hunger"] - 0.01)
        self.state.needs["energy"] = max(0, self.state.needs["energy"] - 0.02)
        self.state.needs["social"] = max(0, self.state.needs["social"] - 0.005)
        
        return {
            "needs": self.state.needs.copy(),
            "locations": environment_state.get("locations", []),
            "nearby_agents": environment_state.get("nearby_agents", [])
        }
    
    def decide(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """Make rule-based decision."""
        needs = perception["needs"]
        
        # Find most pressing need
        most_pressing = min(needs.items(), key=lambda x: x[1])
        need_name, need_level = most_pressing
        
        # Simple rules
        if need_name == "hunger" and need_level < 0.3:
            action = "eat"
            target = "store"
        elif need_name == "energy" and need_level < 0.3:
            action = "rest"
            target = "home"
        elif need_name == "social" and need_level < 0.4:
            action = "socialize"
            target = "park"
        else:
            action = "rest"
            target = "home"
        
        return {
            "action": action,
            "target": target,
            "reasoning": f"Addressing {need_name} need (level: {need_level:.2f})"
        }
    
    def act(self, decision: Dict[str, Any]) -> List[str]:
        """Convert decision to actions and update state."""
        action = decision["action"]
        target = decision.get("target")
        
        # Update state based on action
        if action == "eat":
            self.state.needs["hunger"] = min(1.0, self.state.needs["hunger"] + 0.3)
            self.state.current_activity = "eating"
        elif action == "rest":
            self.state.needs["energy"] = min(1.0, self.state.needs["energy"] + 0.4)
            self.state.current_activity = "resting"
        elif action == "socialize":
            self.state.needs["social"] = min(1.0, self.state.needs["social"] + 0.2)
            self.state.current_activity = "socializing"
        
        # Return action string
        if target:
            return [f"{action}:{target}"]
        return [action]