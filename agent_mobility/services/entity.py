"""Entity wrapper class for simplified navigation system access"""

from typing import List, Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .navigation import AgentMobilitySystem

from ..models.entities import EntityState, PlaceResult, Location
from ..models.enums import TransportMode


class Entity:
    """
    Wrapper class for entity-specific navigation operations.
    Provides convenient access to navigation methods without repeatedly passing entity_id.
    """
    
    def __init__(self, system: 'AgentMobilitySystem', entity_id: str):
        """
        Initialize an Entity instance
        
        Args:
            system: AgentMobilitySystem instance
            entity_id: Unique identifier for this entity
        """
        self._system = system
        self.entity_id = entity_id
    
    @property
    def state(self) -> Optional[EntityState]:
        """Get the current state of this entity"""
        return self._system.get_entity_state(self.entity_id)
    
    @property
    def current_location(self) -> Optional[Location]:
        """Get the current location of this entity"""
        state = self.state
        return state.current_location if state else None
    
    @property
    def destination(self) -> Optional[PlaceResult]:
        """Get the current destination of this entity"""
        state = self.state
        return state.destination if state else None
    
    def search_nearby(
        self,
        query: str,
        radius: int = 5000,
        max_results: int = 20,
        transport_mode: Optional[TransportMode] = None
    ) -> Tuple[List[PlaceResult], str]:
        """
        Search for places near this entity's current location
        
        Args:
            query: Search query (e.g., "restaurants", "coffee shops")
            radius: Search radius in meters
            max_results: Maximum number of results
            transport_mode: Optional specific transport mode to calculate travel times
            
        Returns:
            Tuple of (List of PlaceResult objects with travel times, status string)
        """
        return self._system.search_nearby(
            self.entity_id,
            query,
            radius,
            max_results,
            transport_mode
        )
    
    def get_place_details(self, place_id: str) -> Tuple[Dict, str]:
        """
        Get detailed information about a specific place
        
        Args:
            place_id: Google Maps place ID
            
        Returns:
            Tuple of (Dictionary with place details including reviews, status string)
        """
        return self._system.get_place_details(place_id)
    
    def set_destination(self, place: PlaceResult):
        """
        Set a destination for this entity
        
        Args:
            place: PlaceResult to set as destination
        """
        self._system.set_destination(self.entity_id, place)
    
    def update_location(self, lat: float, lng: float, address: str = None):
        """
        Update this entity's current location
        
        Args:
            lat: Latitude
            lng: Longitude
            address: Optional address string
        """
        self._system.update_location(self.entity_id, lat, lng, address)
    
    def get_search_history(self, limit: int = 50) -> List[Dict]:
        """
        Get search history for this entity
        
        Args:
            limit: Maximum number of history records to return
            
        Returns:
            List of search history records
        """
        return self._system.get_search_history(self.entity_id, limit)
    
    def get_navigation_history(self, limit: int = 50) -> List[Dict]:
        """
        Get navigation history for this entity
        
        Args:
            limit: Maximum number of history records to return
            
        Returns:
            List of navigation history records
        """
        return self._system.get_navigation_history(self.entity_id, limit)
    
    def delete(self):
        """Delete this entity and all associated data"""
        self._system.delete_entity(self.entity_id)
    
    def __repr__(self) -> str:
        """String representation of the entity"""
        state = self.state
        if state:
            loc = state.current_location
            return f"Entity(id='{self.entity_id}', location=({loc.lat}, {loc.lng}))"
        return f"Entity(id='{self.entity_id}', state=None)"
    
    def __str__(self) -> str:
        """Human-readable string representation"""
        return self.__repr__()