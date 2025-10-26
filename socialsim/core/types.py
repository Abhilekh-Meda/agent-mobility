"""
Core type definitions for SocialSim.

This module contains all Pydantic models used throughout the library.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from enum import Enum


class Location(BaseModel):
    """Geographic location coordinates."""
    
    latitude: float = Field(..., ge=-90, le=90, description="Latitude in degrees")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude in degrees")
    
    def __str__(self) -> str:
        return f"({self.latitude:.4f}, {self.longitude:.4f})"


class PersonalityTrait(str, Enum):
    """Big Five personality traits."""
    OPENNESS = "openness"
    CONSCIENTIOUSNESS = "conscientiousness"
    EXTRAVERSION = "extraversion"
    AGREEABLENESS = "agreeableness"
    NEUROTICISM = "neuroticism"


class AgentProfile(BaseModel):
    """Static attributes defining an agent's identity.
    
    These attributes do not change during simulation.
    """
    
    agent_id: str = Field(..., description="Unique identifier for the agent")
    name: str = Field(..., description="Agent's name")
    age: int = Field(..., ge=0, le=120, description="Agent's age in years")
    occupation: str = Field(default="unemployed", description="Agent's occupation")
    personality_traits: Dict[str, float] = Field(
        default_factory=dict,
        description="Personality trait values (0-1 scale)"
    )
    demographic_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional demographic information"
    )
    
    @field_validator('personality_traits')
    @classmethod
    def validate_personality_traits(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Ensure personality traits are in valid range."""
        for trait, value in v.items():
            if not 0 <= value <= 1:
                raise ValueError(f"Personality trait '{trait}' must be between 0 and 1")
        return v
    
    def __str__(self) -> str:
        return f"{self.name} ({self.agent_id}), {self.age}, {self.occupation}"


class AgentState(BaseModel):
    """Dynamic state of an agent that changes during simulation."""
    
    location: Location = Field(
        default_factory=lambda: Location(latitude=0.0, longitude=0.0),
        description="Current geographic location"
    )
    current_activity: str = Field(
        default="idle",
        description="What the agent is currently doing"
    )
    needs: Dict[str, float] = Field(
        default_factory=dict,
        description="Agent's current needs (0-1 scale, 0=desperate, 1=satisfied)"
    )
    emotions: Dict[str, float] = Field(
        default_factory=dict,
        description="Agent's emotional state (0-1 scale)"
    )
    energy: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Agent's energy level"
    )
    inventory: List[str] = Field(
        default_factory=list,
        description="Items the agent possesses"
    )
    relationships: Dict[str, float] = Field(
        default_factory=dict,
        description="Relationship strength with other agents (-1 to 1)"
    )
    
    @field_validator('needs', 'emotions')
    @classmethod
    def validate_scales(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Ensure need/emotion values are in valid range."""
        for key, value in v.items():
            if not 0 <= value <= 1:
                raise ValueError(f"Value for '{key}' must be between 0 and 1")
        return v
    
    @field_validator('relationships')
    @classmethod
    def validate_relationships(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Ensure relationship values are in valid range."""
        for agent_id, strength in v.items():
            if not -1 <= strength <= 1:
                raise ValueError(f"Relationship with '{agent_id}' must be between -1 and 1")
        return v


class SimulationConfig(BaseModel):
    """Configuration for a simulation run."""
    
    name: str = Field(..., description="Name of the simulation")
    start_time: datetime = Field(
        default_factory=datetime.now,
        description="Simulation start time"
    )
    time_step_seconds: int = Field(
        default=60,
        ge=1,
        description="Duration of each simulation step in seconds"
    )
    max_steps: int = Field(
        default=1000,
        ge=1,
        description="Maximum number of steps to run"
    )
    log_interval: int = Field(
        default=10,
        ge=1,
        description="How often to log metrics (in steps)"
    )
    checkpoint_interval: Optional[int] = Field(
        default=None,
        description="How often to save checkpoints (in steps), None to disable"
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility"
    )
    
    def __str__(self) -> str:
        return f"SimulationConfig(name='{self.name}', steps={self.max_steps})"


class ActionType(str, Enum):
    """Types of actions agents can take."""
    MOVE = "move"
    SOCIALIZE = "socialize"
    EAT = "eat"
    WORK = "work"
    REST = "rest"
    CONSUME = "consume"
    MESSAGE = "message"


class Action(BaseModel):
    """Represents an action taken by an agent."""
    
    action_type: ActionType = Field(..., description="Type of action")
    agent_id: str = Field(..., description="Agent performing the action")
    target: Optional[str] = Field(default=None, description="Target of the action")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional action parameters"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the action occurred"
    )
    
    def to_string(self) -> str:
        """Convert action to string format for processing."""
        if self.target:
            return f"{self.action_type.value}:{self.target}"
        return self.action_type.value
    
    @classmethod
    def from_string(cls, action_str: str, agent_id: str) -> "Action":
        """Parse action from string format."""
        parts = action_str.split(":", 1)
        action_type = ActionType(parts[0])
        target = parts[1] if len(parts) > 1 else None
        
        return cls(
            action_type=action_type,
            agent_id=agent_id,
            target=target
        )


class EnvironmentState(BaseModel):
    """State of the environment at a given time."""
    
    time: datetime = Field(default_factory=datetime.now, description="Current time")
    step: int = Field(default=0, description="Current step number")
    locations: List[str] = Field(
        default_factory=list,
        description="Available locations"
    )
    agents_per_location: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Which agents are at each location"
    )
    weather: Optional[str] = Field(
        default=None,
        description="Current weather condition"
    )
    temperature: Optional[float] = Field(
        default=None,
        description="Current temperature"
    )
    
    def get_agents_at(self, location_id: str) -> List[str]:
        """Get list of agent IDs at a specific location."""
        return self.agents_per_location.get(location_id, [])
    
    def get_nearby_agents(self, agent_id: str) -> List[str]:
        """Get agents at the same location as the given agent."""
        for location_id, agents in self.agents_per_location.items():
            if agent_id in agents:
                return [aid for aid in agents if aid != agent_id]
        return []


class LLMConfig(BaseModel):
    """Configuration for LLM provider."""
    
    provider: str = Field(
        default="openai",
        description="LLM provider (openai, anthropic, local)"
    )
    model: str = Field(
        default="gpt-4o-mini",
        description="Model name"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature"
    )
    max_tokens: Optional[int] = Field(
        default=512,
        description="Maximum tokens in response"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key (can also use environment variables)"
    )
    cache_responses: bool = Field(
        default=True,
        description="Whether to cache LLM responses"
    )
    
    def __str__(self) -> str:
        return f"LLMConfig({self.provider}/{self.model})"


class StepMetrics(BaseModel):
    """Metrics collected for a single simulation step."""
    
    step: int = Field(..., description="Step number")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When metrics were collected"
    )
    num_agents: int = Field(..., description="Number of active agents")
    actions_taken: int = Field(default=0, description="Total actions in this step")
    llm_calls: int = Field(default=0, description="Number of LLM API calls")
    step_duration_seconds: float = Field(
        default=0.0,
        description="Time taken to execute step"
    )
    custom_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Custom metrics"
    )
    
    def __str__(self) -> str:
        return (
            f"Step {self.step}: {self.num_agents} agents, "
            f"{self.actions_taken} actions, {self.step_duration_seconds:.2f}s"
        )