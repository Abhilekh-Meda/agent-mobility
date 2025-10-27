"""
SocialSim: Large-Scale LLM-Driven Social Simulation Library

A Python library for simulating societies of LLM-powered agents.
"""

__version__ = "0.1.0"
__author__ = "SocialSim Contributors"
__license__ = "MIT"

from loguru import logger
import sys

# Configure default logging
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO"
)

# Core imports
from socialsim.core.types import (
    AgentProfile,
    AgentState,
    Location,
    SimulationConfig,
    Action,
    ActionType,
    LLMConfig,
)
from socialsim.core.simulation import Simulation

# Agent imports
from socialsim.agents.base import BaseAgent, RandomAgent, SimpleReflexAgent
from socialsim.agents.behaviors.needs import NeedDrivenAgent

# Environment imports
from socialsim.environment.simple import SimpleEnvironment, LocationInfo

# Tools imports
from socialsim.tools.metrics import MetricsCollector

# Make key classes available at package level
__all__ = [
    # Version
    "__version__",
    # Core types
    "AgentProfile",
    "AgentState",
    "Location",
    "SimulationConfig",
    "Action",
    "ActionType",
    "LLMConfig",
    # Core classes
    "Simulation",
    # Agents
    "BaseAgent",
    "RandomAgent",
    "SimpleReflexAgent",
    "NeedDrivenAgent",
    # Environment
    "SimpleEnvironment",
    "LocationInfo",
    # Tools
    "MetricsCollector",
]