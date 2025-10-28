"""
Main simulation orchestrator for SocialSim.

Coordinates agents, environment, and time progression.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import time
from loguru import logger

from socialsim.core.types import SimulationConfig, StepMetrics
from socialsim.agents.base import BaseAgent
from socialsim.environment.simple import SimpleEnvironment
from socialsim.tools.metrics import MetricsCollector


class Simulation:
    """Main simulation orchestrator.
    
    Coordinates:
    - Agent registration and lifecycle
    - Environment state management
    - Action processing
    - Time progression
    - Metrics collection
    - Logging
    
    Phase 1: Sequential, single-threaded execution
    Phase 2: Will add Ray-based distributed execution
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize simulation.
        
        Args:
            name: Simulation name
            config: Optional configuration dictionary
        """
        self.name = name
        
        # Parse config
        config = config or {}
        # create config model (caller may include 'name' in the config dict)
        # Normalize incoming config to a dict and avoid passing 'name' twice
        if config is None:
            cfg = {}
        elif isinstance(config, dict):
            cfg = dict(config)
        elif hasattr(config, "model_dump"):
            cfg = dict(config.model_dump())
        else:
            try:
                cfg = dict(config)
            except Exception:
                cfg = {}
        
        # remove any 'name' provided by caller so we only pass it once
        cfg.pop("name", None)
        self.config = SimulationConfig(name=name, **cfg)
        
        # Core components
        self.agents: Dict[str, BaseAgent] = {}
        self.environment = SimpleEnvironment(
            start_time=self.config.start_time,
            time_step_seconds=self.config.time_step_seconds
        )
        self.metrics = MetricsCollector(simulation_name=name)
        
        # Simulation state
        self.is_running = False
        self.is_paused = False
        self.current_step = 0
        
        # Performance tracking
        self.total_runtime_seconds = 0.0
        self.steps_per_second = 0.0
        
        logger.info(f"Initialized simulation: {name}")
    
    def add_agent(self, agent: BaseAgent, initial_location: str = "home") -> None:
        """Add an agent to the simulation.
        
        Args:
            agent: Agent to add
            initial_location: Starting location
            
        Raises:
            ValueError: If agent ID already exists
        """
        agent_id = agent.profile.agent_id
        
        if agent_id in self.agents:
            raise ValueError(f"Agent '{agent_id}' already exists")
        
        # Register agent
        self.agents[agent_id] = agent
        self.environment.register_agent(agent_id, initial_location)
        
        logger.debug(f"Added agent: {agent.profile.name} ({agent_id})")
    
    def add_agents_batch(
        self,
        agents: List[BaseAgent],
        initial_location: str = "home"
    ) -> None:
        """Add multiple agents efficiently.
        
        Args:
            agents: List of agents to add
            initial_location: Starting location for all agents
        """
        for agent in agents:
            self.add_agent(agent, initial_location)
        
        logger.info(f"Added {len(agents)} agents in batch")
    
    def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent from simulation.
        
        Args:
            agent_id: Agent to remove
            
        Returns:
            True if removed, False if not found
        """
        if agent_id not in self.agents:
            return False
        
        del self.agents[agent_id]
        self.environment.unregister_agent(agent_id)
        
        logger.debug(f"Removed agent: {agent_id}")
        return True
    
    def run(
        self,
        num_steps: Optional[int] = None,
        until_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Run the simulation.
        
        Args:
            num_steps: Number of steps to run (default: from config)
            until_time: Run until this time (alternative to num_steps)
            
        Returns:
            Dictionary with simulation results
            
        Raises:
            ValueError: If simulation already running
        """
        if self.is_running:
            raise ValueError("Simulation is already running")
        
        # Determine how many steps to run
        if num_steps is None and until_time is None:
            num_steps = self.config.max_steps
        elif until_time is not None:
            time_diff = until_time - self.environment.current_time
            num_steps = int(time_diff.total_seconds() / self.config.time_step_seconds)
        
        self.is_running = True
        start_time = time.time()
        
        logger.info(
            f"Starting simulation '{self.name}' for {num_steps} steps "
            f"({len(self.agents)} agents)"
        )
        
        try:
            for step in range(num_steps):
                # Check if paused
                while self.is_paused:
                    time.sleep(0.1)
                
                # Execute one step
                step_metrics = self._execute_step()
                
                # Log progress
                if step % self.config.log_interval == 0:
                    logger.info(
                        f"Step {self.current_step}/{num_steps}: "
                        f"{step_metrics.actions_taken} actions, "
                        f"{step_metrics.llm_calls} LLM calls, "
                        f"{step_metrics.step_duration_seconds:.2f}s"
                    )
        
        except KeyboardInterrupt:
            logger.warning("Simulation interrupted by user")
        
        except Exception as e:
            logger.error(f"Simulation error: {e}")
            raise
        
        finally:
            self.is_running = False
            self.total_runtime_seconds = time.time() - start_time
            self.steps_per_second = self.current_step / max(0.001, self.total_runtime_seconds)
        
        # Generate results
        results = self._generate_results()
        
        logger.info(
            f"Simulation complete: {self.current_step} steps in "
            f"{self.total_runtime_seconds:.1f}s "
            f"({self.steps_per_second:.2f} steps/sec)"
        )
        
        return results
    
    def _execute_step(self) -> StepMetrics:
        """Execute one simulation step.
        
        Returns:
            StepMetrics for this step
        """
        step_start = time.time()
        self.current_step += 1
        
        # Get global environment state
        env_state = self.environment.get_state()
        
        # Track step metrics
        total_actions = 0
        total_llm_calls = 0
        
        # Execute each agent sequentially (Phase 1)
        for agent_id, agent in self.agents.items():
            try:
                # Get agent-specific environment view
                agent_env_state = self.environment.get_state_for_agent(agent_id)
                
                # Agent takes action
                actions = agent.step(agent_env_state)
                
                # Process actions
                self._process_agent_actions(agent_id, actions)
                
                # Track metrics
                total_actions += len(actions)
                total_llm_calls += 1 if hasattr(agent, 'llm_cost_tracker') else 0
                
            except Exception as e:
                logger.error(f"Error executing agent {agent_id}: {e}")
                # Continue with other agents
        
        # Update environment
        self.environment.update()
        
        # Record metrics
        step_duration = time.time() - step_start
        metrics = StepMetrics(
            step=self.current_step,
            num_agents=len(self.agents),
            actions_taken=total_actions,
            llm_calls=total_llm_calls,
            step_duration_seconds=step_duration
        )
        
        self.metrics.record_step(metrics)
        
        return metrics
    
    def _process_agent_actions(self, agent_id: str, actions: List[str]) -> None:
        """Process actions from an agent.
        
        Args:
            agent_id: Agent performing actions
            actions: List of action strings
        """
        for action_str in actions:
            try:
                self._execute_action(agent_id, action_str)
            except Exception as e:
                logger.error(f"Error executing action '{action_str}' for {agent_id}: {e}")
    
    def _execute_action(self, agent_id: str, action_str: str) -> None:
        """Execute a single action.
        
        Args:
            agent_id: Agent performing action
            action_str: Action string in format "action:target"
        """
        # Parse action
        parts = action_str.split(":", 1)
        action_type = parts[0]
        target = parts[1] if len(parts) > 1 else None
        
        # Execute based on action type
        if action_type == "move":
            if target and target in self.environment.locations:
                self.environment.move_agent(agent_id, target)
            else:
                logger.warning(f"Invalid move target '{target}' for {agent_id}")
        
        elif action_type in ["eat", "rest", "work", "socialize", "exercise"]:
            # These are handled by agent's internal state updates
            # In Phase 2+, we'll add resource consumption, etc.
            pass
        
        elif action_type == "message":
            # In Phase 3, we'll implement messaging system
            logger.debug(f"Message action (not yet implemented): {action_str}")
        
        else:
            logger.warning(f"Unknown action type: {action_type}")
    
    def _generate_results(self) -> Dict[str, Any]:
        """Generate simulation results summary.
        
        Returns:
            Dictionary with results
        """
        # Aggregate agent statistics
        total_steps = sum(a.stats["steps_taken"] for a in self.agents.values())
        total_actions = sum(a.stats["actions_taken"] for a in self.agents.values())
        total_llm_calls = sum(a.stats.get("llm_calls", 0) for a in self.agents.values())
        
        # Calculate costs for agents with cost tracking
        total_cost = 0.0
        for agent in self.agents.values():
            if hasattr(agent, 'get_cost_summary'):
                total_cost += agent.get_cost_summary()["estimated_cost_usd"]
        
        return {
            "simulation_name": self.name,
            "steps_completed": self.current_step,
            "runtime_seconds": self.total_runtime_seconds,
            "steps_per_second": self.steps_per_second,
            "num_agents": len(self.agents),
            "total_agent_steps": total_steps,
            "total_actions": total_actions,
            "total_llm_calls": total_llm_calls,
            "estimated_cost_usd": total_cost,
            "environment_stats": self.environment.get_stats(),
            "metrics_summary": self.metrics.get_summary(),
        }
    
    def pause(self) -> None:
        """Pause the simulation."""
        self.is_paused = True
        logger.info("Simulation paused")
    
    def resume(self) -> None:
        """Resume the simulation."""
        self.is_paused = False
        logger.info("Simulation resumed")
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get an agent by ID.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Agent or None if not found
        """
        return self.agents.get(agent_id)
    
    def get_all_agents(self) -> List[BaseAgent]:
        """Get all agents.
        
        Returns:
            List of all agents
        """
        return list(self.agents.values())
    
    def get_agents_by_location(self, location_id: str) -> List[BaseAgent]:
        """Get all agents at a specific location.
        
        Args:
            location_id: Location identifier
            
        Returns:
            List of agents at location
        """
        agent_ids = self.environment.get_agents_at(location_id)
        return [self.agents[aid] for aid in agent_ids if aid in self.agents]
    
    def reset(self) -> None:
        """Reset simulation to initial state.
        
        Keeps agents and locations, but resets time and statistics.
        """
        self.current_step = 0
        self.is_running = False
        self.is_paused = False
        self.total_runtime_seconds = 0.0
        
        # Reset environment
        self.environment.reset()
        
        # Re-register all agents
        for agent_id, agent in self.agents.items():
            agent.reset()
            self.environment.register_agent(agent_id, "home")
        
        # Reset metrics
        self.metrics = MetricsCollector(simulation_name=self.name)
        
        logger.info("Simulation reset")
    
    def save_results(self, filepath: Optional[str] = None) -> str:
        """Save simulation results to file.
        
        Args:
            filepath: Output file path (default: auto-generated)
            
        Returns:
            Path where results were saved
        """
        return self.metrics.save(filepath)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current simulation status.
        
        Returns:
            Dictionary with status information
        """
        return {
            "name": self.name,
            "is_running": self.is_running,
            "is_paused": self.is_paused,
            "current_step": self.current_step,
            "num_agents": len(self.agents),
            "current_time": self.environment.current_time.isoformat(),
            "runtime_seconds": self.total_runtime_seconds,
            "steps_per_second": self.steps_per_second
        }
    
    def __str__(self) -> str:
        return (
            f"Simulation(name='{self.name}', "
            f"agents={len(self.agents)}, "
            f"step={self.current_step})"
        )
    
    def __repr__(self) -> str:
        return f"<Simulation '{self.name}' at step {self.current_step}>"