"""
Checkpointing system for Phase 2.

Save and restore complete simulation state at any point.
"""

import gzip
import pickle
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from loguru import logger

from socialsim import __version__


class CheckpointManager:
    """Save and restore simulation state.
    
    Features:
    - Complete state capture
    - Compression (gzip)
    - Version compatibility
    - Incremental saves
    - Automatic cleanup
    """
    
    def __init__(
        self,
        checkpoint_dir: str = './checkpoints',
        compression_level: int = 6,
        keep_last_n: int = 10
    ):
        """Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for checkpoints
            compression_level: Gzip compression level (1-9)
            keep_last_n: Number of checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.compression_level = compression_level
        self.keep_last_n = keep_last_n
        
        logger.info(f"CheckpointManager initialized at {self.checkpoint_dir}")
    
    def save_checkpoint(
        self,
        simulation: Any,
        checkpoint_name: Optional[str] = None,
        include_metrics: bool = True
    ) -> str:
        """Save complete simulation state.
        
        Args:
            simulation: Simulation instance to save
            checkpoint_name: Custom name (default: auto-generated)
            include_metrics: Whether to include metrics data
            
        Returns:
            Path to saved checkpoint
        """
        # Generate checkpoint name
        if checkpoint_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_name = f"{simulation.name}_step_{simulation.current_step}_{timestamp}"
        
        logger.info(f"Saving checkpoint: {checkpoint_name}")
        start_time = datetime.now()
        
        # Gather simulation state
        checkpoint_data = {
            'metadata': {
                'simulation_name': simulation.name,
                'step': simulation.current_step,
                'timestamp': datetime.now().isoformat(),
                'num_agents': len(simulation.agents),
                'version': __version__,
                'checkpoint_name': checkpoint_name
            },
            'config': simulation.config.dict() if hasattr(simulation.config, 'dict') else simulation.config,
            'agents': self._serialize_agents(simulation.agents),
            'environment': self._serialize_environment(simulation.environment),
            'simulation_state': {
                'is_running': simulation.is_running,
                'is_paused': simulation.is_paused,
                'current_step': simulation.current_step,
                'total_runtime_seconds': simulation.total_runtime_seconds,
                'steps_per_second': simulation.steps_per_second
            }
        }
        
        # Optionally include metrics
        if include_metrics and hasattr(simulation, 'metrics'):
            checkpoint_data['metrics'] = self._serialize_metrics(simulation.metrics)
        
        # Save to file
        filepath = self.checkpoint_dir / f"{checkpoint_name}.pkl.gz"
        
        with gzip.open(filepath, 'wb', compresslevel=self.compression_level) as f:
            pickle.dump(checkpoint_data, f)
        
        # Log stats
        duration = (datetime.now() - start_time).total_seconds()
        file_size_mb = filepath.stat().st_size / 1024 / 1024
        
        logger.info(
            f"Checkpoint saved: {file_size_mb:.2f} MB in {duration:.2f}s "
            f"({file_size_mb/duration:.2f} MB/s)"
        )
        
        # Cleanup old checkpoints
        self.cleanup_old_checkpoints()
        
        return str(filepath)
    
    def load_checkpoint(
        self,
        filepath: str,
        restore_metrics: bool = True
    ) -> Dict[str, Any]:
        """Load simulation state from checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            restore_metrics: Whether to restore metrics
            
        Returns:
            Checkpoint data dictionary
            
        Raises:
            FileNotFoundError: If checkpoint doesn't exist
            ValueError: If checkpoint is invalid or incompatible
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        logger.info(f"Loading checkpoint: {filepath.name}")
        start_time = datetime.now()
        
        # Load from file
        with gzip.open(filepath, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        # Validate checkpoint
        self._validate_checkpoint(checkpoint_data)
        
        # Check version compatibility
        checkpoint_version = checkpoint_data['metadata'].get('version', '0.0.0')
        if not self._is_compatible_version(checkpoint_version):
            logger.warning(
                f"Checkpoint version {checkpoint_version} may not be compatible "
                f"with current version {__version__}"
            )
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Checkpoint loaded in {duration:.2f}s")
        
        return checkpoint_data
    
    def restore_simulation(
        self,
        filepath: str,
        simulation_class: type
    ) -> Any:
        """Restore complete simulation from checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            simulation_class: Simulation class to instantiate
            
        Returns:
            Restored simulation instance
        """
        checkpoint_data = self.load_checkpoint(filepath)
        
        # Create new simulation instance
        sim = simulation_class(
            name=checkpoint_data['metadata']['simulation_name'],
            config=checkpoint_data['config']
        )
        
        # Restore agents
        sim.agents = self._deserialize_agents(
            checkpoint_data['agents'],
            sim  # Pass simulation for context
        )
        
        # Restore environment
        sim.environment = self._deserialize_environment(
            checkpoint_data['environment'],
            sim.environment.__class__
        )
        
        # Restore simulation state
        state = checkpoint_data['simulation_state']
        sim.current_step = state['current_step']
        sim.total_runtime_seconds = state['total_runtime_seconds']
        sim.steps_per_second = state['steps_per_second']
        
        # Restore metrics if present
        if 'metrics' in checkpoint_data and hasattr(sim, 'metrics'):
            sim.metrics = self._deserialize_metrics(checkpoint_data['metrics'])
        
        logger.info(
            f"Simulation restored: {len(sim.agents)} agents at step {sim.current_step}"
        )
        
        return sim
    
    def _serialize_agents(self, agents: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize agents dictionary.
        
        Args:
            agents: Dictionary of agents
            
        Returns:
            Serialized agents data
        """
        return {
            agent_id: agent.to_dict()
            for agent_id, agent in agents.items()
        }
    
    def _deserialize_agents(
        self,
        agents_data: Dict[str, Any],
        simulation: Any
    ) -> Dict[str, Any]:
        """Deserialize agents dictionary.
        
        Args:
            agents_data: Serialized agents data
            simulation: Simulation instance for context
            
        Returns:
            Dictionary of restored agents
        """
        from socialsim.agents.base import RandomAgent, SimpleReflexAgent
        from socialsim.agents.behaviors.needs import NeedDrivenAgent
        
        # Map class names to classes
        agent_classes = {
            'RandomAgent': RandomAgent,
            'SimpleReflexAgent': SimpleReflexAgent,
            'NeedDrivenAgent': NeedDrivenAgent
        }
        
        agents = {}
        
        for agent_id, agent_data in agents_data.items():
            class_name = agent_data.get('class_name', 'RandomAgent')
            agent_class = agent_classes.get(class_name, RandomAgent)
            
            # Get LLM config from simulation config
            # Handle both dict and Pydantic model cases
            if isinstance(simulation.config, dict):
                config_dict = simulation.config
            else:
                # Convert Pydantic model to dict
                config_dict = simulation.config.dict() if hasattr(simulation.config, 'dict') else simulation.config.model_dump()
            
            llm_config = config_dict.get('llm_config', {
                'provider': 'openai',
                'model': 'gpt-4o-mini'
            })
            
            # Restore agent
            agent = agent_class.from_dict(agent_data, llm_config)
            agents[agent_id] = agent
        
        return agents
    
    def _serialize_environment(self, environment: Any) -> Dict[str, Any]:
        """Serialize environment state.
        
        Args:
            environment: Environment instance
            
        Returns:
            Serialized environment data
        """
        return {
            'class_name': environment.__class__.__name__,
            'locations': {
                loc_id: {
                    'location_id': loc.location_id,
                    'name': loc.name,
                    'location_type': loc.location_type,
                    'capacity': loc.capacity,
                    'properties': loc.properties
                }
                for loc_id, loc in environment.locations.items()
            },
            'agent_locations': dict(environment.agent_locations),
            'agents_at_location': {
                loc_id: list(agents)
                for loc_id, agents in environment.agents_at_location.items()
            },
            'current_time': environment.current_time.isoformat(),
            'current_step': environment.current_step,
            'stats': environment.stats.copy()
        }
    
    def _deserialize_environment(
        self,
        env_data: Dict[str, Any],
        env_class: type
    ) -> Any:
        """Deserialize environment.
        
        Args:
            env_data: Serialized environment data
            env_class: Environment class to instantiate
            
        Returns:
            Restored environment instance
        """
        from socialsim.environment.simple import SimpleEnvironment, LocationInfo
        from datetime import datetime
        
        # Create new environment
        env = SimpleEnvironment()
        
        # Restore locations
        for loc_id, loc_data in env_data['locations'].items():
            env.locations[loc_id] = LocationInfo(**loc_data)
            env.agents_at_location[loc_id] = set()
        
        # Restore agent locations
        env.agent_locations = env_data['agent_locations'].copy()
        
        # Restore agents at locations
        for loc_id, agent_list in env_data['agents_at_location'].items():
            env.agents_at_location[loc_id] = set(agent_list)
        
        # Restore time and state
        env.current_time = datetime.fromisoformat(env_data['current_time'])
        env.current_step = env_data['current_step']
        env.stats = env_data['stats'].copy()
        
        return env
    
    def _serialize_metrics(self, metrics: Any) -> Dict[str, Any]:
        """Serialize metrics data.
        
        Args:
            metrics: Metrics instance
            
        Returns:
            Serialized metrics data
        """
        return {
            'simulation_name': metrics.simulation_name,
            'step_metrics': [m.dict() for m in metrics.step_metrics],
            'custom_metrics': metrics.custom_metrics.copy(),
            'start_time': metrics.start_time.isoformat(),
            'end_time': metrics.end_time.isoformat() if metrics.end_time else None
        }
    
    def _deserialize_metrics(self, metrics_data: Dict[str, Any]) -> Any:
        """Deserialize metrics.
        
        Args:
            metrics_data: Serialized metrics data
            
        Returns:
            Restored metrics instance
        """
        from socialsim.tools.metrics import MetricsCollector
        from socialsim.core.types import StepMetrics
        from datetime import datetime
        
        metrics = MetricsCollector(metrics_data['simulation_name'])
        
        # Restore step metrics
        metrics.step_metrics = [
            StepMetrics(**m) for m in metrics_data['step_metrics']
        ]
        
        # Restore custom metrics
        metrics.custom_metrics = metrics_data['custom_metrics'].copy()
        
        # Restore timestamps
        metrics.start_time = datetime.fromisoformat(metrics_data['start_time'])
        if metrics_data['end_time']:
            metrics.end_time = datetime.fromisoformat(metrics_data['end_time'])
        
        return metrics
    
    def _validate_checkpoint(self, checkpoint_data: Dict[str, Any]):
        """Validate checkpoint data.
        
        Args:
            checkpoint_data: Checkpoint to validate
            
        Raises:
            ValueError: If checkpoint is invalid
        """
        required_keys = ['metadata', 'config', 'agents', 'environment']
        
        for key in required_keys:
            if key not in checkpoint_data:
                raise ValueError(f"Invalid checkpoint: missing '{key}'")
        
        # Validate metadata
        metadata = checkpoint_data['metadata']
        if 'version' not in metadata:
            logger.warning("Checkpoint missing version information")
    
    def _is_compatible_version(self, checkpoint_version: str) -> bool:
        """Check if checkpoint version is compatible.
        
        Args:
            checkpoint_version: Version of checkpoint
            
        Returns:
            True if compatible
        """
        # For now, simple major version check
        current_major = __version__.split('.')[0]
        checkpoint_major = checkpoint_version.split('.')[0]
        
        return current_major == checkpoint_major
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List available checkpoints.
        
        Returns:
            List of checkpoint information
        """
        checkpoints = []
        
        for filepath in sorted(self.checkpoint_dir.glob("*.pkl.gz")):
            try:
                # Quick peek at metadata without full load
                with gzip.open(filepath, 'rb') as f:
                    # Load only first KB to get metadata
                    data = pickle.load(f)
                    metadata = data.get('metadata', {})
                
                checkpoints.append({
                    'name': filepath.stem,
                    'path': str(filepath),
                    'size_mb': filepath.stat().st_size / 1024 / 1024,
                    'created': datetime.fromtimestamp(filepath.stat().st_mtime),
                    'simulation_name': metadata.get('simulation_name', 'unknown'),
                    'step': metadata.get('step', 0),
                    'num_agents': metadata.get('num_agents', 0),
                    'version': metadata.get('version', 'unknown')
                })
                
            except Exception as e:
                logger.error(f"Error reading checkpoint {filepath}: {e}")
        
        return checkpoints
    
    def cleanup_old_checkpoints(self, keep_last_n: Optional[int] = None):
        """Delete old checkpoints, keeping only the most recent.
        
        Args:
            keep_last_n: Number to keep (default: from init)
        """
        keep_n = keep_last_n or self.keep_last_n
        
        checkpoints = sorted(
            self.checkpoint_dir.glob("*.pkl.gz"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        if len(checkpoints) > keep_n:
            to_delete = checkpoints[keep_n:]
            
            for filepath in to_delete:
                try:
                    filepath.unlink()
                    logger.debug(f"Deleted old checkpoint: {filepath.name}")
                except Exception as e:
                    logger.error(f"Error deleting {filepath}: {e}")
            
            logger.info(f"Cleaned up {len(to_delete)} old checkpoints")
    
    def delete_checkpoint(self, checkpoint_name: str) -> bool:
        """Delete a specific checkpoint.
        
        Args:
            checkpoint_name: Name of checkpoint to delete
            
        Returns:
            Success status
        """
        filepath = self.checkpoint_dir / f"{checkpoint_name}.pkl.gz"
        
        if not filepath.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_name}")
            return False
        
        try:
            filepath.unlink()
            logger.info(f"Deleted checkpoint: {checkpoint_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting checkpoint: {e}")
            return False
    
    def get_checkpoint_info(self, checkpoint_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a checkpoint without loading it.
        
        Args:
            checkpoint_name: Name of checkpoint
            
        Returns:
            Checkpoint information or None
        """
        checkpoints = self.list_checkpoints()
        
        for checkpoint in checkpoints:
            if checkpoint['name'] == checkpoint_name:
                return checkpoint
        
        return None