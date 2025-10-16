"""
Agent Mobility System
A scalable system for AI agents to search, choose, and navigate to destinations.
"""

from .services import AgentMobilitySystem, Entity


__version__ = "1.0.0"
__author__ = "Agent Mobility Team"

# Update __all__ to reflect what you are exposing from the top-level package
__all__ = [
    'AgentMobilitySystem',
    'Entity',
    # 'DatabaseManager',
    # 'Location',
    # 'PlaceResult',
    # 'EntityState',
    # 'TransportMode'
]