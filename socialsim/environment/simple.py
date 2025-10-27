"""
Simple environment implementation for Phase 1.

Provides basic location management and agent tracking without complex spatial calculations.
"""

from typing import Dict, List, Set, Optional, Any
from datetime import datetime, timedelta
from loguru import logger
from pydantic import BaseModel, Field

from socialsim.core.types import Location, EnvironmentState


class LocationInfo(BaseModel):
    """Information about a location in the environment."""
    
    location_id: str = Field(..., description="Unique identifier")
    name: str = Field(..., description="Human-readable name")
    location_type: str = Field(..., description="Type: residential, commercial, recreation, etc.")
    capacity: int = Field(default=100, description="Maximum number of agents")
    coordinates: Optional[Location] = Field(default=None, description="Geographic coordinates")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Additional properties")
    
    def is_full(self, current_count: int) -> bool:
        """Check if location is at capacity."""
        return current_count >= self.capacity
    
    def __str__(self) -> str:
        return f"{self.name} ({self.location_type})"


class SimpleEnvironment:
    """Simple environment for Phase 1 simulations.
    
    Features:
    - Named locations with types
    - Agent position tracking
    - Time management
    - Basic capacity limits
    - State queries
    
    Does NOT include (coming in later phases):
    - Real geographic maps
    - Complex spatial calculations
    - Transportation networks
    - Weather systems
    """
    
    def __init__(
        self,
        start_time: Optional[datetime] = None,
        time_step_seconds: int = 60
    ):
        """Initialize environment.
        
        Args:
            start_time: Simulation start time (default: now)
            time_step_seconds: Duration of each step in seconds
        """
        self.locations: Dict[str, LocationInfo] = {}
        self.agents_at_location: Dict[str, Set[str]] = {}
        self.agent_locations: Dict[str, str] = {}  # agent_id -> location_id
        
        # Time management
        self.current_time = start_time or datetime.now()
        self.time_step = timedelta(seconds=time_step_seconds)
        self.current_step = 0
        
        # Statistics
        self.stats = {
            "total_movements": 0,
            "total_locations": 0,
            "total_agents": 0
        }
        
        logger.info("Initialized SimpleEnvironment")
    
    def add_location(
        self,
        location_id: str,
        name: str,
        location_type: str,
        capacity: int = 100,
        coordinates: Optional[Location] = None,
        properties: Optional[Dict[str, Any]] = None
    ) -> LocationInfo:
        """Add a location to the environment.
        
        Args:
            location_id: Unique identifier
            name: Human-readable name
            location_type: Type of location (residential, commercial, etc.)
            capacity: Maximum number of agents
            coordinates: Optional geographic coordinates
            properties: Additional properties
            
        Returns:
            LocationInfo object
            
        Raises:
            ValueError: If location_id already exists
        """
        if location_id in self.locations:
            raise ValueError(f"Location '{location_id}' already exists")
        
        location = LocationInfo(
            location_id=location_id,
            name=name,
            location_type=location_type,
            capacity=capacity,
            coordinates=coordinates,
            properties=properties or {}
        )
        
        self.locations[location_id] = location
        self.agents_at_location[location_id] = set()
        self.stats["total_locations"] += 1
        
        logger.debug(f"Added location: {name} ({location_id})")
        return location
    
    def remove_location(self, location_id: str) -> bool:
        """Remove a location from the environment.
        
        Args:
            location_id: Location to remove
            
        Returns:
            True if removed, False if not found
            
        Raises:
            ValueError: If location has agents
        """
        if location_id not in self.locations:
            return False
        
        if self.agents_at_location[location_id]:
            raise ValueError(
                f"Cannot remove location '{location_id}': "
                f"{len(self.agents_at_location[location_id])} agents present"
            )
        
        del self.locations[location_id]
        del self.agents_at_location[location_id]
        self.stats["total_locations"] -= 1
        
        logger.debug(f"Removed location: {location_id}")
        return True
    
    def register_agent(self, agent_id: str, initial_location: str = "home") -> None:
        """Register an agent in the environment.
        
        Args:
            agent_id: Agent identifier
            initial_location: Starting location (default: "home")
            
        Raises:
            ValueError: If agent already registered or location doesn't exist
        """
        if agent_id in self.agent_locations:
            raise ValueError(f"Agent '{agent_id}' already registered")
        
        # Create "home" location if it doesn't exist and is requested
        if initial_location == "home" and "home" not in self.locations:
            self.add_location("home", "Home", "residential", capacity=1000) #TODO: adjust these weird no reason numbers
        
        if initial_location not in self.locations:
            raise ValueError(f"Location '{initial_location}' does not exist")
        
        self.agent_locations[agent_id] = initial_location
        self.agents_at_location[initial_location].add(agent_id)
        self.stats["total_agents"] += 1
        
        logger.debug(f"Registered agent {agent_id} at {initial_location}")
    
    def unregister_agent(self, agent_id: str) -> bool:
        """Remove an agent from the environment.
        
        Args:
            agent_id: Agent to remove
            
        Returns:
            True if removed, False if not found
        """
        if agent_id not in self.agent_locations:
            return False
        
        # Remove from location
        current_location = self.agent_locations[agent_id]
        self.agents_at_location[current_location].discard(agent_id)
        
        # Remove from tracking
        del self.agent_locations[agent_id]
        self.stats["total_agents"] -= 1
        
        logger.debug(f"Unregistered agent {agent_id}")
        return True
    
    def move_agent(
        self,
        agent_id: str,
        to_location: str,
        force: bool = False
    ) -> bool:
        """Move agent to a new location.
        
        Args:
            agent_id: Agent to move
            to_location: Destination location
            force: If True, ignore capacity limits
            
        Returns:
            True if move succeeded, False otherwise
            
        Raises:
            ValueError: If agent or location doesn't exist
        """
        if agent_id not in self.agent_locations:
            raise ValueError(f"Agent '{agent_id}' not registered")
        
        if to_location not in self.locations:
            raise ValueError(f"Location '{to_location}' does not exist")
        
        # Check capacity
        location_info = self.locations[to_location]
        current_count = len(self.agents_at_location[to_location])
        
        if not force and location_info.is_full(current_count):
            logger.warning(
                f"Cannot move {agent_id} to {to_location}: at capacity "
                f"({current_count}/{location_info.capacity})"
            )
            return False
        
        # Remove from old location
        from_location = self.agent_locations[agent_id]
        self.agents_at_location[from_location].discard(agent_id)
        
        # Add to new location
        self.agent_locations[agent_id] = to_location
        self.agents_at_location[to_location].add(agent_id)
        self.stats["total_movements"] += 1
        
        logger.debug(f"Moved {agent_id}: {from_location} -> {to_location}")
        return True
    
    def get_agent_location(self, agent_id: str) -> Optional[str]:
        """Get agent's current location.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Location ID or None if not found
        """
        return self.agent_locations.get(agent_id)
    
    def get_agents_at(self, location_id: str) -> List[str]:
        """Get all agents at a location.
        
        Args:
            location_id: Location identifier
            
        Returns:
            List of agent IDs
        """
        return list(self.agents_at_location.get(location_id, set()))
    
    def get_nearby_agents(
        self,
        agent_id: str,
        include_self: bool = False
    ) -> List[str]:
        """Get agents at the same location as the given agent.
        
        Args:
            agent_id: Reference agent
            include_self: If True, include agent_id in results
            
        Returns:
            List of agent IDs at same location
        """
        location = self.get_agent_location(agent_id)
        if not location:
            return []
        
        agents = self.get_agents_at(location)
        
        if not include_self and agent_id in agents:
            agents.remove(agent_id)
        
        return agents
    
    def get_location_info(self, location_id: str) -> Optional[LocationInfo]:
        """Get information about a location.
        
        Args:
            location_id: Location identifier
            
        Returns:
            LocationInfo or None if not found
        """
        return self.locations.get(location_id)
    
    def get_all_locations(self) -> List[str]:
        """Get all location IDs.
        
        Returns:
            List of location IDs
        """
        return list(self.locations.keys())
    
    def get_locations_by_type(self, location_type: str) -> List[str]:
        """Get all locations of a specific type.
        
        Args:
            location_type: Type to filter by
            
        Returns:
            List of location IDs matching type
        """
        return [
            loc_id for loc_id, loc_info in self.locations.items()
            if loc_info.location_type == location_type
        ]
    
    def get_state(self) -> EnvironmentState:
        """Get current environment state.
        
        Returns:
            EnvironmentState object with current state
        """
        return EnvironmentState(
            time=self.current_time,
            step=self.current_step,
            locations=self.get_all_locations(),
            agents_per_location={
                loc_id: self.get_agents_at(loc_id)
                for loc_id in self.locations.keys()
            }
        )
    
    def get_state_for_agent(self, agent_id: str) -> Dict[str, Any]:
        """Get environment state from agent's perspective.
        
        Args:
            agent_id: Agent requesting state
            
        Returns:
            Dictionary with agent-specific environment view
        """
        current_location = self.get_agent_location(agent_id)
        nearby_agents = self.get_nearby_agents(agent_id)
        
        # Get all locations (in Phase 2+, filter by distance)
        locations = self.get_all_locations()
        
        return {
            "time": self.current_time,
            "step": self.current_step,
            "current_location": current_location,
            "locations": locations,
            "nearby_agents": nearby_agents,
            "agents_at_location": len(nearby_agents) + 1,  # Include self
        }
    
    def update(self) -> None:
        """Advance environment by one time step.
        
        Updates:
        - Current time
        - Step counter
        - Any time-based environment dynamics (in later phases)
        """
        self.current_time += self.time_step
        self.current_step += 1
        
        logger.debug(
            f"Environment step {self.current_step}: "
            f"{self.current_time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
    
    def reset(self) -> None:
        """Reset environment to initial state.
        
        Clears all agents but keeps locations.
        """
        # Clear agent tracking
        for location_id in self.agents_at_location:
            self.agents_at_location[location_id].clear()
        self.agent_locations.clear()
        
        # Reset time
        self.current_time = datetime.now()
        self.current_step = 0
        
        # Reset stats
        self.stats["total_movements"] = 0
        self.stats["total_agents"] = 0
        # Keep total_locations
        
        logger.info("Environment reset")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get environment statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            **self.stats,
            "locations_by_type": {
                loc_type: len(self.get_locations_by_type(loc_type))
                for loc_type in set(
                    loc.location_type for loc in self.locations.values()
                )
            },
            "average_occupancy": (
                sum(len(agents) for agents in self.agents_at_location.values()) /
                max(1, len(self.locations))
            ),
            "current_step": self.current_step,
            "current_time": self.current_time.isoformat()
        }
    
    def __str__(self) -> str:
        return (
            f"SimpleEnvironment(locations={len(self.locations)}, "
            f"agents={len(self.agent_locations)}, step={self.current_step})"
        )
    
    def __repr__(self) -> str:
        return f"<SimpleEnvironment step={self.current_step}>"