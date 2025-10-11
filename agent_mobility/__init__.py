"""
Agent Mobility System
A scalable system for AI agents to search, choose, and navigate to destinations.
"""

from .models import Location, TravelInfo, PlaceResult, EntityState, TransportMode
from .navigation import NavigationSystem
from .database import DatabaseManager

__version__ = "1.0.0"
__author__ = "Agent Mobility Team"

__all__ = [
    'NavigationSystem',
    'DatabaseManager', 
    'Location',
    'TravelInfo', 
    'PlaceResult',
    'EntityState',
    'TransportMode'
]
