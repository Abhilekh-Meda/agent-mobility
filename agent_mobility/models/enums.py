"""Enums for the agent mobility system"""

from enum import Enum


class TransportMode(Enum):
    """Available transport modes"""
    DRIVING = "driving"
    WALKING = "walking"
    BICYCLING = "bicycling"
    TRANSIT = "transit"
