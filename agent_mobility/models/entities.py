"""Data entities for the agent mobility system"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional


@dataclass
class Location:
    """Represents a geographic location"""
    lat: float
    lng: float
    address: Optional[str] = None


@dataclass
class TravelInfo:
    """Travel information for a specific mode"""
    mode: str
    duration_minutes: int
    distance_meters: int
    duration_text: str
    distance_text: str


@dataclass
class PlaceResult:
    """Represents a place from search results"""
    place_id: str
    name: str
    address: str
    location: Location
    rating: Optional[float]
    user_ratings_total: Optional[int]
    types: List[str]
    price_level: Optional[int]
    open_now: Optional[bool]
    travel_times: Dict[str, TravelInfo]
    
    def to_dict(self):
        """Convert to dictionary for serialization"""
        d = asdict(self)
        d['location'] = asdict(self.location)
        return d


@dataclass
class EntityState:
    """State of an entity in the navigation system"""
    entity_id: str
    current_location: Location
    destination: Optional[PlaceResult]
    last_updated: str
    
    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {
            'entity_id': self.entity_id,
            'current_location': asdict(self.current_location),
            'destination': self.destination.to_dict() if self.destination else None,
            'last_updated': self.last_updated
        }
