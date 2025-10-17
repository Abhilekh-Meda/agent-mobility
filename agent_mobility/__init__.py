"""
Agent Mobility System
A scalable system for AI agents to search, choose, and navigate to destinations.
"""

from .services import AgentMobilitySystem, Entity
from .models import Location, TravelInfo, PlaceResult, EntityState, TransportMode


__version__ = "1.0.0"
__author__ = "Agent Mobility Team"

# Update __all__ to reflect what is being exposed to top level
__all__ = [
    'AgentMobilitySystem',
    'Entity',
    'DatabaseManager',
    'Location',
    'PlaceResult',
    'EntityState',
    'TransportMode'
]