from typing import Dict, Any, List, Tuple
from pydantic import BaseModel, Field
from datetime import datetime

class Location(BaseModel):
    """Geographic location"""
    latitude: float
    longitude: float
    
class AgentProfile(BaseModel):
    """Static agent attributes"""
    agent_id: str
    name: str
    age: int
    occupation: str
    personality_traits: Dict[str, float] = Field(default_factory=dict)
    
class AgentState(BaseModel):
    """Dynamic agent state"""
    location: Location
    current_activity: str = "idle"
    needs: Dict[str, float] = Field(default_factory=dict)
    emotions: Dict[str, float] = Field(default_factory=dict)
    energy: float = 1.0
    
class SimulationConfig(BaseModel):
    """Simulation configuration"""
    name: str
    start_time: datetime
    time_step_seconds: int = 60
    max_steps: int = 1000