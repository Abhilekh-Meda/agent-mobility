"""
Need-driven agent implementation using Maslow's hierarchy of needs.

This agent uses LLM-based reasoning to make decisions based on its current needs.
"""

from typing import Dict, Any, List, Optional
import json
from loguru import logger
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from socialsim.agents.base import BaseAgent
from socialsim.core.types import AgentProfile


class AgentDecision(BaseModel):
    """Structured output for agent decisions."""
    
    action: str = Field(description="The action to take: move, socialize, eat, work, rest")
    target: Optional[str] = Field(default=None, description="Target location or agent")
    reasoning: str = Field(description="Brief explanation of why this action was chosen")
    priority: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Priority of this action (0-1)"
    )


class NeedDrivenAgent(BaseAgent):
    """Agent driven by Maslow's hierarchy of needs with LLM-based decision making.
    
    This agent maintains five levels of needs from Maslow's hierarchy:
    - Physiological: Basic survival needs (food, water, rest)
    - Safety: Security, health, stability
    - Belonging: Social connections, relationships
    - Esteem: Achievement, recognition, respect
    - Self-actualization: Personal growth, creativity
    
    The agent uses an LLM to make intelligent decisions based on:
    - Current need levels
    - Available options in environment
    - Personality traits
    - Recent experiences
    """
    
    def __init__(
        self,
        profile: AgentProfile,
        llm_config: Dict[str, Any],
        memory_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize need-driven agent.
        
        Args:
            profile: Agent profile with personality traits
            llm_config: LLM configuration
            memory_config: Optional memory configuration
        """
        super().__init__(profile, llm_config, memory_config)
        
        # Initialize Maslow's hierarchy of needs
        self.state.needs = {
            "physiological": 1.0,  # Food, water, rest
            "safety": 1.0,         # Security, health
            "belonging": 1.0,      # Social connections
            "esteem": 1.0,         # Achievement, recognition
            "self_actualization": 1.0  # Personal growth
        }
        
        # Need decay rates (per step)
        self.need_decay_rates = {
            "physiological": 0.015,  # Decays fastest
            "safety": 0.005,
            "belonging": 0.008,
            "esteem": 0.003,
            "self_actualization": 0.002  # Decays slowest
        }
        
        # Prompt template for LLM decision making
        self.decision_prompt = self._create_decision_prompt()
        
        # Output parser for structured responses
        self.output_parser = JsonOutputParser(pydantic_object=AgentDecision)
        
        # LLM call tracking
        self.llm_cost_tracker = {
            "total_calls": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0
        }
        
        logger.info(f"Initialized NeedDrivenAgent: {self.profile.name}")
    
    def _create_decision_prompt(self) -> ChatPromptTemplate:
        """Create the prompt template for LLM decision making."""
        
        system_message = """You are simulating a person's decision-making process in a social simulation.
You must help this person decide what to do based on their current needs and situation.

The person has needs based on Maslow's hierarchy:
- Physiological (0-1): Food, water, rest. Low = desperate for basic needs
- Safety (0-1): Security, health. Low = feeling unsafe or unhealthy  
- Belonging (0-1): Social connection. Low = lonely, isolated
- Esteem (0-1): Achievement, recognition. Low = feeling unaccomplished
- Self-actualization (0-1): Personal growth. Low = unfulfilled potential

Consider their personality traits when making decisions.
Be realistic - people prioritize lower-level needs first (physiological > safety > belonging > esteem > self-actualization).

Choosing to do nothing is also an option.

{format_instructions}"""

        user_message = """Current Situation:
Person: {name}, {age} years old, {occupation}
Personality: {personality}

Current Needs (0=desperate, 1=satisfied):
- Physiological: {physiological:.2f}
- Safety: {safety:.2f}
- Belonging: {belonging:.2f}
- Esteem: {esteem:.2f}
- Self-actualization: {self_actualization:.2f}

Available Locations: {locations}
Nearby People: {nearby_people}
Current Time: {time}
Current Activity: {current_activity}

What should this person do next? Respond with a JSON object containing:
- action: One of [move, socialize, eat, work, rest, exercise]
- target: Location or person name (if applicable)
- reasoning: Brief explanation (one sentence)
- priority: How urgent this action is (0.0 to 1.0)"""

        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("user", user_message)
        ])
    
    def perceive(self, environment_state: Dict[str, Any]) -> Dict[str, Any]:
        """Perceive environment and update internal needs.
        
        Args:
            environment_state: Current environment state
            
        Returns:
            Perception dictionary with filtered/processed information
        """
        # Decay all needs over time
        for need, decay_rate in self.need_decay_rates.items():
            self.state.needs[need] = max(0.0, self.state.needs[need] - decay_rate)
        
        # Extract relevant information from environment
        perception = {
            "needs": self.state.needs.copy(),
            "locations": environment_state.get("locations", []),
            "nearby_agents": environment_state.get("nearby_agents", []),
            "time": environment_state.get("time"),
            "step": environment_state.get("step", 0),
            "current_activity": self.state.current_activity,
            "energy": self.state.energy
        }
        
        # Update energy based on activity
        if self.state.current_activity == "resting":
            self.state.energy = min(1.0, self.state.energy + 0.1)
        else:
            self.state.energy = max(0.0, self.state.energy - 0.02)
        
        # Add context from recent memory
        recent_memories = self.get_recent_memory(n=3)
        perception["recent_actions"] = [
            m.get("actions", []) for m in recent_memories
        ]
        
        return perception
    
    def decide(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """Make decision using LLM based on current needs and perception.
        
        Args:
            perception: Output from perceive()
            
        Returns:
            Decision dictionary with action, target, reasoning
        """
        needs = perception["needs"]
        
        # Find most pressing need (Maslow's hierarchy - prioritize lower levels)
        need_weights = {
            "physiological": 5.0,
            "safety": 4.0,
            "belonging": 3.0,
            "esteem": 2.0,
            "self_actualization": 1.0
        }
        
        weighted_needs = {
            need: (1.0 - level) * need_weights[need]
            for need, level in needs.items()
        }
        most_pressing_need = max(weighted_needs.items(), key=lambda x: x[1])
        
        # Check if we should use LLM or fall back to rules
        use_llm = self._should_use_llm(needs)
        
        if use_llm:
            try:
                decision = self._llm_decide(perception)
                self.stats["llm_calls"] += 1
                self.llm_cost_tracker["total_calls"] += 1
                # Rough token estimate: ~500 tokens per call
                #TODO: change this based on how much it actually costs
                self.llm_cost_tracker["total_tokens"] += 500
                # GPT-4o-mini: ~$0.00015 per 1K tokens (input) + $0.0006 per 1K tokens (output)
                self.llm_cost_tracker["estimated_cost_usd"] += 0.0004
            except Exception as e:
                logger.warning(
                    f"LLM decision failed for {self.profile.agent_id}: {e}. "
                    f"Falling back to rule-based."
                )
                decision = self._rule_based_decide(needs, perception)
        else:
            decision = self._rule_based_decide(needs, perception)
        
        # Add metadata
        decision["most_pressing_need"] = most_pressing_need[0]
        decision["need_level"] = needs[most_pressing_need[0]]
        
        return decision
    
    def _should_use_llm(self, needs: Dict[str, float]) -> bool:
        """Determine if we should use LLM or fall back to rules.
        
        Use LLM when:
        - Any need is critically low (< 0.2) - emergency situations use rules
        - Every 3rd decision (to reduce costs) TODO: lets see about this
        
        Args:
            needs: Current need levels
            
        Returns:
            True if should use LLM
        """
        # Emergency situations: use fast rules
        if any(level < 0.2 for level in needs.values()):
            return False
        
        # Use LLM for 1 in 3 decisions to reduce cost
        return self.stats["decisions_made"] % 3 == 0
    
    def _llm_decide(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to make decision.
        
        Args:
            perception: Current perception
            
        Returns:
            Decision dictionary
        """
        needs = perception["needs"]
        
        # Format personality traits for prompt
        personality_str = ", ".join([
            f"{trait}: {value:.2f}"
            for trait, value in self.profile.personality_traits.items()
        ])
        
        # Format nearby people
        nearby_str = ", ".join(perception["nearby_agents"][:5]) if perception["nearby_agents"] else "None"
        
        # Create prompt
        prompt = self.decision_prompt.partial(
            format_instructions=self.output_parser.get_format_instructions()
        )
        
        #TODO: make this into ReAct framework prompt
        # Invoke LLM
        chain = prompt | self.llm | self.output_parser
        
        response = chain.invoke({
            "name": self.profile.name,
            "age": self.profile.age,
            "occupation": self.profile.occupation,
            "personality": personality_str or "Not specified",
            "physiological": needs["physiological"],
            "safety": needs["safety"],
            "belonging": needs["belonging"],
            "esteem": needs["esteem"],
            "self_actualization": needs["self_actualization"],
            "locations": ", ".join(perception["locations"][:10]),
            "nearby_people": nearby_str,
            "time": str(perception.get("time", "unknown")),
            "current_activity": perception["current_activity"]
        })
        
        # Convert AgentDecision to dict
        if isinstance(response, dict):
            return response
        else:
            return {
                "action": response.get("action", "rest"),
                "target": response.get("target"),
                "reasoning": response.get("reasoning", "LLM response"),
                "priority": response.get("priority", 0.5)
            }
    
    def _rule_based_decide(
        self,
        needs: Dict[str, float],
        perception: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fall back to simple rule-based decision making.
        
        Args:
            needs: Current needs
            perception: Current perception
            
        Returns:
            Decision dictionary
        """
        # Find lowest need
        lowest_need = min(needs.items(), key=lambda x: x[1])
        need_name, need_level = lowest_need
        
        # Simple rules for each need type
        if need_name == "physiological" and need_level < 0.4:
            if "store" in perception["locations"] or "restaurant" in perception["locations"]:
                target = "store" if "store" in perception["locations"] else "restaurant"
                return {
                    "action": "eat",
                    "target": target,
                    "reasoning": f"Physiological need critically low ({need_level:.2f})",
                    "priority": 1.0 - need_level
                }
            else:
                return {
                    "action": "rest",
                    "target": "home",
                    "reasoning": "Need rest, no food available",
                    "priority": 0.8
                }
        
        elif need_name == "safety" and need_level < 0.4:
            return {
                "action": "rest",
                "target": "home",
                "reasoning": f"Safety need low ({need_level:.2f}), seeking shelter",
                "priority": 0.9
            }
        
        elif need_name == "belonging" and need_level < 0.5:
            if perception["nearby_agents"]:
                return {
                    "action": "socialize",
                    "target": perception["nearby_agents"][0],
                    "reasoning": f"Social connection needed ({need_level:.2f})",
                    "priority": 0.7
                }
            elif "park" in perception["locations"] or "cafe" in perception["locations"]:
                target = "park" if "park" in perception["locations"] else "cafe"
                return {
                    "action": "move",
                    "target": target,
                    "reasoning": "Seeking social interaction",
                    "priority": 0.6
                }
        
        elif need_name == "esteem" and need_level < 0.5:
            if "work" in perception["locations"] or "gym" in perception["locations"]:
                target = "work" if "work" in perception["locations"] else "gym"
                return {
                    "action": "move",
                    "target": target,
                    "reasoning": "Seeking achievement",
                    "priority": 0.5
                }
        
        # Default: rest if energy low, otherwise explore
        if self.state.energy < 0.3:
            return {
                "action": "rest",
                "target": "home",
                "reasoning": "Low energy",
                "priority": 0.7
            }
        else:
            locations = perception["locations"]
            target = locations[self.stats["steps_taken"] % len(locations)] if locations else "home"
            return {
                "action": "move",
                "target": target,
                "reasoning": "Exploring environment",
                "priority": 0.3
            }
    
    #TODO: this is  very, very basic, will make it more robust later
    def act(self, decision: Dict[str, Any]) -> List[str]:
        """Convert decision into executable actions and update state.
        
        Args:
            decision: Decision from decide()
            
        Returns:
            List of action strings
        """
        action = decision["action"]
        target = decision.get("target")
        
        # Update state based on action
        self.state.current_activity = action
        
        if action == "eat":
            self.state.needs["physiological"] = min(1.0, self.state.needs["physiological"] + 0.4)
            self.state.energy = min(1.0, self.state.energy + 0.1)
            
        elif action == "rest":
            self.state.needs["physiological"] = min(1.0, self.state.needs["physiological"] + 0.2)
            self.state.needs["safety"] = min(1.0, self.state.needs["safety"] + 0.2)
            self.state.energy = min(1.0, self.state.energy + 0.3)
            
        elif action == "socialize":
            self.state.needs["belonging"] = min(1.0, self.state.needs["belonging"] + 0.3)
            self.state.needs["esteem"] = min(1.0, self.state.needs["esteem"] + 0.1)
            
        elif action == "work":
            self.state.needs["esteem"] = min(1.0, self.state.needs["esteem"] + 0.2)
            self.state.needs["self_actualization"] = min(1.0, self.state.needs["self_actualization"] + 0.1)
            self.state.energy = max(0.0, self.state.energy - 0.1)
            
        elif action == "exercise":
            self.state.needs["safety"] = min(1.0, self.state.needs["safety"] + 0.1)
            self.state.needs["esteem"] = min(1.0, self.state.needs["esteem"] + 0.15)
            self.state.energy = max(0.0, self.state.energy - 0.15)
        
        # Build action string
        if target:
            action_str = f"{action}:{target}"
        else:
            action_str = action
        
        return [action_str]
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get LLM cost tracking summary.
        
        Returns:
            Dictionary with cost information
        """
        return {
            **self.llm_cost_tracker,
            "cost_per_call": (
                self.llm_cost_tracker["estimated_cost_usd"] / 
                max(1, self.llm_cost_tracker["total_calls"])
            )
        }