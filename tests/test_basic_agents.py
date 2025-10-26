"""
Basic example demonstrating agent creation and simple interactions.

This example shows how to:
1. Create agent profiles
2. Initialize different agent types
3. Run agents through perceive-decide-act cycle
4. Track agent behavior over time

This is a minimal example that doesn't require API keys (uses SimpleReflexAgent).
"""

import sys
from pathlib import Path

import os
from dotenv import load_dotenv

load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from socialsim import AgentProfile, RandomAgent, SimpleReflexAgent
from datetime import datetime


def create_sample_agents():
    """Create a variety of sample agents."""
    
    agents = []
    
    # Create 5 different agents with varying profiles
    profiles = [
        {
            "agent_id": "alice",
            "name": "Alice",
            "age": 28,
            "occupation": "software_engineer",
            "personality_traits": {
                "extraversion": 0.7,
                "conscientiousness": 0.8
            }
        },
        {
            "agent_id": "bob",
            "name": "Bob",
            "age": 35,
            "occupation": "teacher",
            "personality_traits": {
                "extraversion": 0.5,
                "agreeableness": 0.9
            }
        },
        {
            "agent_id": "carol",
            "name": "Carol",
            "age": 42,
            "occupation": "artist",
            "personality_traits": {
                "openness": 0.9,
                "neuroticism": 0.4
            }
        },
        {
            "agent_id": "dave",
            "name": "Dave",
            "age": 31,
            "occupation": "retail_worker",
            "personality_traits": {
                "conscientiousness": 0.6,
                "agreeableness": 0.7
            }
        },
        {
            "agent_id": "eve",
            "name": "Eve",
            "age": 25,
            "occupation": "student",
            "personality_traits": {
                "openness": 0.8,
                "extraversion": 0.6
            }
        }
    ]
    
    llm_config = {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "temperature": 0.7
    }
    
    for profile_data in profiles:
        profile = AgentProfile(**profile_data)
        # Use SimpleReflexAgent (no API key needed)
        agent = SimpleReflexAgent(profile, llm_config)
        agents.append(agent)
    
    return agents


def simulate_environment_state(step: int):
    """Create mock environment state for testing."""
    return {
        "step": step,
        "time": datetime.now(),
        "locations": ["home", "work", "store", "park", "gym"],
        "nearby_agents": ["alice", "bob", "carol"] if step % 3 == 0 else [],
        "weather": "sunny" if step % 2 == 0 else "cloudy"
    }


def run_basic_simulation():
    """Run a basic simulation with sample agents."""
    
    print("=" * 60)
    print("SocialSim - Basic Agent Example")
    print("=" * 60)
    print()
    
    # Create agents
    print("Creating agents...")
    agents = create_sample_agents()
    print(f"Created {len(agents)} agents:")
    for agent in agents:
        print(f"  - {agent.profile.name} ({agent.profile.occupation})")
    print()
    
    # Run simulation for 20 steps
    num_steps = 20
    print(f"Running simulation for {num_steps} steps...")
    print()
    
    for step in range(num_steps):
        print(f"--- Step {step + 1}/{num_steps} ---")
        
        # Get environment state
        env_state = simulate_environment_state(step)
        
        # Each agent takes a step
        for agent in agents:
            actions = agent.step(env_state)
            
            # Print agent action
            needs = agent.state.needs
            most_pressing = min(needs.items(), key=lambda x: x[1])
            
            print(
                f"{agent.profile.name:8s} | "
                f"Activity: {agent.state.current_activity:12s} | "
                f"Pressing need: {most_pressing[0]} ({most_pressing[1]:.2f}) | "
                f"Actions: {actions}"
            )
        
        print()
        
        # Print summary every 5 steps
        if (step + 1) % 5 == 0:
            print("📊 Summary:")
            total_actions = sum(a.stats["actions_taken"] for a in agents)
            avg_energy = sum(a.state.energy for a in agents) / len(agents)
            
            print(f"  Total actions taken: {total_actions}")
            print(f"  Average energy level: {avg_energy:.2f}")
            
            # Show need levels
            print("  Current needs:")
            for agent in agents:
                hunger = agent.state.needs.get("hunger", 1.0)
                print(f"    {agent.profile.name}: hunger={hunger:.2f}")
            print()
    
    print("=" * 60)
    print("Simulation Complete!")
    print("=" * 60)
    print()
    
    # Print final statistics
    print("📈 Final Statistics:")
    print()
    for agent in agents:
        stats = agent.get_stats()
        print(f"{agent.profile.name}:")
        print(f"  Steps taken: {stats['steps_taken']}")
        print(f"  Actions taken: {stats['actions_taken']}")
        print(f"  Decisions made: {stats['decisions_made']}")
        print(f"  Memory size: {stats['memory_size']}")
        print(f"  Final activity: {stats['current_activity']}")
        print(f"  Final energy: {stats['energy']:.2f}")
        print()


def demonstrate_agent_types():
    """Demonstrate different agent types."""
    
    print("=" * 60)
    print("Agent Types Demonstration")
    print("=" * 60)
    print()
    
    profile = AgentProfile(
        agent_id="demo_agent",
        name="Demo Agent",
        age=30,
        occupation="demonstrator"
    )
    
    llm_config = {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "temperature": 0.7
    }
    
    env_state = simulate_environment_state(0)
    
    # 1. RandomAgent
    print("1. RandomAgent - Makes random decisions")
    random_agent = RandomAgent(profile, llm_config)
    
    for i in range(3):
        actions = random_agent.step(env_state)
        print(f"   Step {i+1}: {actions}")
    print()
    
    # 2. SimpleReflexAgent
    print("2. SimpleReflexAgent - Rule-based decisions")
    reflex_agent = SimpleReflexAgent(profile, llm_config)
    
    # Set different need levels to see different behaviors
    test_cases = [
        ("High hunger", {"hunger": 0.2, "energy": 1.0, "social": 1.0}),
        ("Low energy", {"hunger": 1.0, "energy": 0.2, "social": 1.0}),
        ("Low social", {"hunger": 1.0, "energy": 1.0, "social": 0.3}),
    ]
    
    for case_name, needs in test_cases:
        reflex_agent.state.needs = needs
        actions = reflex_agent.step(env_state)
        print(f"   {case_name}: {actions}")
    
    print()


def demonstrate_memory():
    """Demonstrate agent memory system."""
    
    print("=" * 60)
    print("Agent Memory Demonstration")
    print("=" * 60)
    print()
    
    profile = AgentProfile(
        agent_id="memory_agent",
        name="Memory Agent",
        age=30,
        occupation="memorizer"
    )
    
    llm_config = {
        "provider": "openai",
        "model": "gpt-4o-mini"
    }
    
    agent = RandomAgent(profile, llm_config)
    agent.memory_config = {"max_size": 5}  # Small memory for demo
    
    print(f"Agent memory limit: {agent.memory_config['max_size']}")
    print()
    
    # Take several steps
    for step in range(10):
        env_state = simulate_environment_state(step)
        agent.step(env_state)
    
    print(f"Total steps taken: {agent.stats['steps_taken']}")
    print(f"Current memory size: {len(agent.memory)}")
    print()
    
    # Show recent memories
    print("Recent memories (last 3):")
    recent = agent.get_recent_memory(n=3)
    for i, memory in enumerate(recent, 1):
        print(f"  {i}. Step {memory['step']}: "
              f"Action={memory['actions'][0] if memory['actions'] else 'none'}")
    
    print()


if __name__ == "__main__":
    print()
    print("🌍 Welcome to SocialSim!")
    print()
    
    # Run demonstrations
    try:
        # Main simulation
        run_basic_simulation()
        
        # Agent types demo
        demonstrate_agent_types()
        
        # Memory demo
        demonstrate_memory()
        
        print("✅ All demonstrations completed successfully!")
        print()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Simulation interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()