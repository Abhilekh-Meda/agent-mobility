"""
Ray-based distributed execution engine for SocialSim Phase 2.

Enables parallel agent execution across multiple workers for 10,000+ agent simulations.
"""

import ray
from ray.util.queue import Queue
from typing import Dict, List, Any, Optional
import asyncio
from loguru import logger
import psutil
import time

from socialsim.agents.base import BaseAgent
from socialsim.core.types import AgentProfile


@ray.remote
class RayAgent:
    """Ray actor wrapper for agents.
    
    Each RayAgent runs in a separate process/worker and can execute
    agent steps independently.
    """
    
    def __init__(
        self,
        agent_data: Dict[str, Any],
        llm_config: Dict[str, Any]
    ):
        """Initialize remote agent.
        
        Args:
            agent_data: Serialized agent state
            llm_config: LLM configuration
        """
        # Reconstruct agent from serialized data
        agent_class_name = agent_data['class_name']
        
        # Import agent class dynamically
        if agent_class_name == 'NeedDrivenAgent':
            from socialsim.agents.behaviors.needs import NeedDrivenAgent
            agent_class = NeedDrivenAgent
        elif agent_class_name == 'SimpleReflexAgent':
            from socialsim.agents.base import SimpleReflexAgent
            agent_class = SimpleReflexAgent
        else:
            from socialsim.agents.base import RandomAgent
            agent_class = RandomAgent
        
        # Restore agent
        profile = AgentProfile(**agent_data['profile'])
        self.agent = agent_class(profile, llm_config)
        
        # Restore state
        if 'state' in agent_data:
            self.agent.state = agent_data['state']
        if 'memory' in agent_data:
            self.agent.memory = agent_data['memory']
        if 'stats' in agent_data:
            self.agent.stats = agent_data['stats']
        
        self.agent_id = profile.agent_id
        logger.debug(f"Initialized RayAgent: {self.agent_id}")
    
    def step(self, environment_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one agent step.
        
        Args:
            environment_state: Current environment state
            
        Returns:
            Dictionary with actions and updated state
        """
        try:
            actions = self.agent.step(environment_state)
            
            return {
                'agent_id': self.agent_id,
                'actions': actions,
                'state': self.agent.state.dict(),
                'stats': self.agent.stats,
                'success': True
            }
        except Exception as e:
            logger.error(f"Error in agent {self.agent_id}: {e}")
            return {
                'agent_id': self.agent_id,
                'actions': [],
                'error': str(e),
                'success': False
            }
    
    def get_state(self) -> Dict[str, Any]:
        """Get current agent state.
        
        Returns:
            Serialized agent state
        """
        return {
            'agent_id': self.agent_id,
            'profile': self.agent.profile.dict(),
            'state': self.agent.state.dict(),
            'memory': self.agent.memory[-100:],  # Last 100 only
            'stats': self.agent.stats
        }
    
    def update_state(self, state_update: Dict[str, Any]) -> bool:
        """Update agent state.
        
        Args:
            state_update: State changes to apply
            
        Returns:
            Success status
        """
        try:
            for key, value in state_update.items():
                if hasattr(self.agent.state, key):
                    setattr(self.agent.state, key, value)
            return True
        except Exception as e:
            logger.error(f"Error updating state for {self.agent_id}: {e}")
            return False


class RayExecutionEngine:
    """Distributed execution engine using Ray.
    
    Coordinates parallel execution of agents across Ray cluster workers.
    """
    
    def __init__(
        self,
        num_workers: Optional[int] = None,
        resources_per_worker: Optional[Dict[str, float]] = None
    ):
        """Initialize Ray execution engine.
        
        Args:
            num_workers: Number of Ray workers (default: CPU count)
            resources_per_worker: Resource allocation per worker
        """
        self.num_workers = num_workers or psutil.cpu_count()
        self.resources_per_worker = resources_per_worker or {'num_cpus': 1}
        
        # Initialize Ray if not already running
        if not ray.is_initialized():
            ray.init(
                num_cpus=self.num_workers,
                ignore_reinit_error=True,
                logging_level=logging.WARNING
            )
            logger.info(f"Initialized Ray with {self.num_workers} workers")
        
        # Agent actor registry
        self.ray_agents: Dict[str, ray.ObjectRef] = {}
        self.agent_partitions: List[List[str]] = []
        
        # Performance tracking
        self.stats = {
            'total_steps': 0,
            'total_agent_steps': 0,
            'parallel_execution_time': 0.0,
            'avg_step_time': 0.0
        }
        
        logger.info("RayExecutionEngine initialized")
    
    def register_agents(
        self,
        agents: Dict[str, BaseAgent],
        llm_config: Dict[str, Any]
    ) -> None:
        """Register agents as Ray actors.
        
        Args:
            agents: Dictionary of agents to register
            llm_config: LLM configuration for agents
        """
        logger.info(f"Registering {len(agents)} agents as Ray actors...")
        
        start_time = time.time()
        
        # Serialize agents
        agent_data_list = []
        for agent_id, agent in agents.items():
            agent_data = {
                'class_name': agent.__class__.__name__,
                'profile': agent.profile.dict(),
                'state': agent.state.dict(),
                'memory': agent.memory[-100:],  # Last 100 memories
                'stats': agent.stats
            }
            agent_data_list.append((agent_id, agent_data))
        
        # Create Ray actors in parallel
        futures = []
        for agent_id, agent_data in agent_data_list:
            future = RayAgent.remote(agent_data, llm_config)
            futures.append((agent_id, future))
        
        # Store actor references
        for agent_id, future in futures:
            self.ray_agents[agent_id] = future
        
        # Partition agents for load balancing
        self._partition_agents()
        
        duration = time.time() - start_time
        logger.info(
            f"Registered {len(agents)} agents in {duration:.2f}s "
            f"({len(agents)/duration:.0f} agents/sec)"
        )
    
    def _partition_agents(self) -> None:
        """Partition agents across workers for load balancing."""
        agent_ids = list(self.ray_agents.keys())
        partition_size = len(agent_ids) // self.num_workers + 1
        
        self.agent_partitions = [
            agent_ids[i:i + partition_size]
            for i in range(0, len(agent_ids), partition_size)
        ]
        
        logger.debug(
            f"Partitioned {len(agent_ids)} agents into "
            f"{len(self.agent_partitions)} partitions"
        )
    
    async def execute_step_parallel(
        self,
        environment_states: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Execute one step for all agents in parallel.
        
        Args:
            environment_states: Environment state for each agent
            
        Returns:
            Dictionary mapping agent_id to results
        """
        start_time = time.time()
        self.stats['total_steps'] += 1
        
        # Submit all agent steps to Ray
        futures = {}
        for agent_id, ray_agent in self.ray_agents.items():
            env_state = environment_states.get(agent_id, {})
            future = ray_agent.step.remote(env_state)
            futures[agent_id] = future
        
        # Collect results
        results = {}
        completed = 0
        total = len(futures)
        
        for agent_id, future in futures.items():
            try:
                result = ray.get(future)
                results[agent_id] = result
                completed += 1
                
                # Log progress every 1000 agents
                if completed % 1000 == 0:
                    logger.debug(f"Completed {completed}/{total} agents")
                    
            except Exception as e:
                logger.error(f"Error getting result for {agent_id}: {e}")
                results[agent_id] = {
                    'agent_id': agent_id,
                    'actions': [],
                    'error': str(e),
                    'success': False
                }
        
        # Update stats
        duration = time.time() - start_time
        self.stats['parallel_execution_time'] += duration
        self.stats['total_agent_steps'] += len(results)
        self.stats['avg_step_time'] = (
            self.stats['parallel_execution_time'] / 
            self.stats['total_steps']
        )
        
        logger.debug(
            f"Parallel step completed: {len(results)} agents in {duration:.2f}s "
            f"({len(results)/duration:.0f} agents/sec)"
        )
        
        return results
    
    def get_agent_states(
        self,
        agent_ids: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Get current state of agents.
        
        Args:
            agent_ids: Specific agents to query (default: all)
            
        Returns:
            Dictionary mapping agent_id to state
        """
        if agent_ids is None:
            agent_ids = list(self.ray_agents.keys())
        
        futures = {
            agent_id: self.ray_agents[agent_id].get_state.remote()
            for agent_id in agent_ids
        }
        
        states = {}
        for agent_id, future in futures.items():
            try:
                states[agent_id] = ray.get(future)
            except Exception as e:
                logger.error(f"Error getting state for {agent_id}: {e}")
        
        return states
    
    def update_agent_states(
        self,
        state_updates: Dict[str, Dict[str, Any]]
    ) -> Dict[str, bool]:
        """Update states for multiple agents.
        
        Args:
            state_updates: Dictionary mapping agent_id to state updates
            
        Returns:
            Dictionary mapping agent_id to success status
        """
        futures = {}
        for agent_id, update in state_updates.items():
            if agent_id in self.ray_agents:
                future = self.ray_agents[agent_id].update_state.remote(update)
                futures[agent_id] = future
        
        results = {}
        for agent_id, future in futures.items():
            try:
                results[agent_id] = ray.get(future)
            except Exception as e:
                logger.error(f"Error updating {agent_id}: {e}")
                results[agent_id] = False
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics.
        
        Returns:
            Dictionary with performance stats
        """
        return {
            **self.stats,
            'num_workers': self.num_workers,
            'num_agents': len(self.ray_agents),
            'num_partitions': len(self.agent_partitions),
            'ray_cluster_resources': ray.cluster_resources()
        }
    
    def shutdown(self) -> None:
        """Shutdown Ray and cleanup resources."""
        logger.info("Shutting down RayExecutionEngine...")
        
        # Clear actor references
        self.ray_agents.clear()
        self.agent_partitions.clear()
        
        # Shutdown Ray
        if ray.is_initialized():
            ray.shutdown()
        
        logger.info("RayExecutionEngine shut down")
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.shutdown()
        except:
            pass


class RayResourceMonitor:
    """Monitor Ray cluster resources and performance."""
    
    @staticmethod
    def get_cluster_info() -> Dict[str, Any]:
        """Get Ray cluster information.
        
        Returns:
            Cluster status and resources
        """
        if not ray.is_initialized():
            return {'error': 'Ray not initialized'}
        
        return {
            'cluster_resources': ray.cluster_resources(),
            'available_resources': ray.available_resources(),
            'nodes': ray.nodes(),
            'is_initialized': ray.is_initialized()
        }
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get memory usage statistics.
        
        Returns:
            Memory usage in MB
        """
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent()
        }
    
    @staticmethod
    def log_resource_usage():
        """Log current resource usage."""
        cluster_info = RayResourceMonitor.get_cluster_info()
        memory_info = RayResourceMonitor.get_memory_usage()
        
        logger.info("=" * 60)
        logger.info("Ray Cluster Resources:")
        if 'error' not in cluster_info:
            logger.info(f"  Total CPUs: {cluster_info['cluster_resources'].get('CPU', 0)}")
            logger.info(f"  Available CPUs: {cluster_info['available_resources'].get('CPU', 0)}")
            logger.info(f"  Nodes: {len(cluster_info['nodes'])}")
        logger.info(f"Memory Usage:")
        logger.info(f"  RSS: {memory_info['rss_mb']:.1f} MB")
        logger.info(f"  Percent: {memory_info['percent']:.1f}%")
        logger.info("=" * 60)