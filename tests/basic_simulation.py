"""
Complete simulation example with 100 LLM-powered agents.

This demonstrates all Phase 1 features:
- Need-driven agents with LLM decision making
- Multi-location environment
- Metrics collection and analysis
- Agent diversity (personalities, occupations, ages)
"""

import sys
import random
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from socialsim.core.simulation import Simulation
from socialsim.core.types import AgentProfile, SimulationConfig
from socialsim.agents.behaviors.needs import NeedDrivenAgent
from loguru import logger

import os
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO")


def create_diverse_agents(num_agents: int, llm_config: dict) -> list:
    """Create a diverse population of agents.
    
    Args:
        num_agents: Number of agents to create
        llm_config: LLM configuration
        
    Returns:
        List of NeedDrivenAgent instances
    """
    # Occupation distribution
    occupations = [
        ("software_engineer", 0.15),
        ("teacher", 0.12),
        ("healthcare_worker", 0.10),
        ("retail_worker", 0.10),
        ("artist", 0.08),
        ("accountant", 0.08),
        ("construction_worker", 0.07),
        ("student", 0.15),
        ("retired", 0.10),
        ("unemployed", 0.05)
    ]
    
    # First names for variety
    first_names = [
        "Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Henry",
        "Iris", "Jack", "Kate", "Leo", "Maria", "Noah", "Olivia", "Peter",
        "Quinn", "Rachel", "Sam", "Tara", "Uma", "Victor", "Wendy", "Xavier",
        "Yara", "Zack", "Amy", "Ben", "Chloe", "Dan"
    ]
    
    agents = []
    
    for i in range(num_agents):
        # Generate diverse attributes
        name = f"{random.choice(first_names)} {chr(65 + i % 26)}"
        age = random.randint(18, 75)
        
        # Select occupation based on distribution
        occupation = random.choices(
            [occ for occ, _ in occupations],
            weights=[weight for _, weight in occupations]
        )[0]
        
        # Generate Big Five personality traits
        personality_traits = {
            "openness": random.uniform(0.3, 0.9),
            "conscientiousness": random.uniform(0.3, 0.9),
            "extraversion": random.uniform(0.2, 0.9),
            "agreeableness": random.uniform(0.3, 0.9),
            "neuroticism": random.uniform(0.1, 0.7)
        }
        
        # Create profile
        profile = AgentProfile(
            agent_id=f"agent_{i:03d}",
            name=name,
            age=age,
            occupation=occupation,
            personality_traits=personality_traits,
            demographic_info={
                "cohort": i % 10,  # Group into cohorts
                "index": i
            }
        )
        
        # Create agent
        agent = NeedDrivenAgent(profile, llm_config)
        agents.append(agent)
    
    return agents


def setup_urban_environment(sim: Simulation) -> None:
    """Set up a realistic urban environment with various location types.
    
    Args:
        sim: Simulation instance
    """
    # Residential
    sim.environment.add_location(
        "home", "Home", "residential",
        capacity=1000,
        properties={"comfort": 0.9, "safety": 0.9}
    )
    
    sim.environment.add_location(
        "apartment_complex", "Apartment Complex", "residential",
        capacity=200,
        properties={"comfort": 0.7, "safety": 0.8}
    )
    
    # Commercial
    sim.environment.add_location(
        "grocery_store", "Joe's Grocery", "commercial",
        capacity=50,
        properties={"opens": 7, "closes": 22}
    )
    
    sim.environment.add_location(
        "shopping_mall", "City Mall", "commercial",
        capacity=200,
        properties={"opens": 10, "closes": 21}
    )
    
    sim.environment.add_location(
        "restaurant", "Downtown Cafe", "commercial",
        capacity=40,
        properties={"opens": 8, "closes": 23, "food_quality": 0.8}
    )
    
    # Workplaces
    sim.environment.add_location(
        "office_building", "Tech Tower", "workplace",
        capacity=150,
        properties={"opens": 8, "closes": 18}
    )
    
    sim.environment.add_location(
        "hospital", "City Hospital", "workplace",
        capacity=100,
        properties={"opens": 0, "closes": 24}  # 24/7
    )
    
    sim.environment.add_location(
        "school", "Central School", "workplace",
        capacity=80,
        properties={"opens": 7, "closes": 16}
    )
    
    # Recreation
    sim.environment.add_location(
        "park", "Central Park", "recreation",
        capacity=300,
        properties={"outdoor": True, "activities": ["walking", "sports"]}
    )
    
    sim.environment.add_location(
        "gym", "Fitness Center", "recreation",
        capacity=60,
        properties={"opens": 6, "closes": 22}
    )
    
    sim.environment.add_location(
        "library", "Public Library", "recreation",
        capacity=100,
        properties={"opens": 9, "closes": 20, "quiet": True}
    )
    
    sim.environment.add_location(
        "cinema", "Movie Theater", "recreation",
        capacity=200,
        properties={"opens": 12, "closes": 24}
    )
    
    logger.info(f"Created {len(sim.environment.locations)} locations")


def print_simulation_header():
    """Print a nice header for the simulation."""
    print("\n" + "=" * 70)
    print("  🌍 SOCIALSIM - Large-Scale Social Simulation")
    print("  Phase 1 Complete Example: 100 LLM-Powered Agents")
    print("=" * 70)
    print()


def print_progress_update(sim: Simulation, step: int, total_steps: int):
    """Print periodic progress updates.
    
    Args:
        sim: Simulation instance
        step: Current step
        total_steps: Total steps
    """
    print(f"\n📊 Progress Update - Step {step}/{total_steps}")
    print("-" * 70)
    
    # Agent activity summary
    activities = {}
    for agent in sim.get_all_agents():
        activity = agent.state.current_activity
        activities[activity] = activities.get(activity, 0) + 1
    
    print("Current Activities:")
    for activity, count in sorted(activities.items(), key=lambda x: -x[1]):
        print(f"  {activity:20s}: {count:3d} agents")
    
    # Location occupancy
    print("\nLocation Occupancy:")
    for loc_id, loc_info in list(sim.environment.locations.items())[:5]:
        count = len(sim.environment.get_agents_at(loc_id))
        occupancy_pct = (count / loc_info.capacity) * 100
        print(f"  {loc_info.name:25s}: {count:3d}/{loc_info.capacity:3d} ({occupancy_pct:5.1f}%)")
    
    # Average needs
    avg_needs = {
        "physiological": 0.0,
        "safety": 0.0,
        "belonging": 0.0,
        "esteem": 0.0,
        "self_actualization": 0.0
    }
    
    for agent in sim.get_all_agents():
        if hasattr(agent.state, 'needs'):
            for need, value in agent.state.needs.items():
                if need in avg_needs:
                    avg_needs[need] += value
    
    num_agents = len(sim.get_all_agents())
    for need in avg_needs:
        avg_needs[need] /= num_agents
    
    print("\nAverage Need Levels (0=desperate, 1=satisfied):")
    for need, value in avg_needs.items():
        bar_length = int(value * 30)
        bar = "█" * bar_length + "░" * (30 - bar_length)
        print(f"  {need:20s}: {bar} {value:.2f}")
    
    print()


def analyze_results(sim: Simulation, results: dict):
    """Analyze and display simulation results.
    
    Args:
        sim: Simulation instance
        results: Results dictionary from simulation
    """
    print("\n" + "=" * 70)
    print("  📈 SIMULATION RESULTS")
    print("=" * 70)
    print()
    
    # Overall statistics
    print("Overall Statistics:")
    print(f"  Simulation Name:        {results['simulation_name']}")
    print(f"  Steps Completed:        {results['steps_completed']}")
    print(f"  Runtime:                {results['runtime_seconds']:.1f} seconds")
    print(f"  Performance:            {results['steps_per_second']:.2f} steps/sec")
    print(f"  Total Agents:           {results['num_agents']}")
    print(f"  Total Actions:          {results['total_actions']:,}")
    print(f"  Total LLM Calls:        {results['total_llm_calls']:,}")
    print(f"  Estimated Cost:         ${results['estimated_cost_usd']:.2f}")
    print()
    
    # Agent statistics
    print("Agent Statistics:")
    all_agents = sim.get_all_agents()
    
    # Activity distribution
    activities = {}
    for agent in all_agents:
        activity = agent.state.current_activity
        activities[activity] = activities.get(activity, 0) + 1
    
    print("  Final Activities:")
    for activity, count in sorted(activities.items(), key=lambda x: -x[1]):
        pct = (count / len(all_agents)) * 100
        print(f"    {activity:20s}: {count:3d} ({pct:5.1f}%)")
    
    # Average steps and actions per agent
    avg_steps = sum(a.stats["steps_taken"] for a in all_agents) / len(all_agents)
    avg_actions = sum(a.stats["actions_taken"] for a in all_agents) / len(all_agents)
    
    print(f"\n  Average per Agent:")
    print(f"    Steps taken:          {avg_steps:.1f}")
    print(f"    Actions taken:        {avg_actions:.1f}")
    print()
    
    # Environment statistics
    env_stats = results['environment_stats']
    print("Environment Statistics:")
    print(f"  Total Locations:        {env_stats['total_locations']}")
    print(f"  Total Movements:        {env_stats['total_movements']:,}")
    print(f"  Average Occupancy:      {env_stats['average_occupancy']:.1f} agents/location")
    print()
    
    # LLM cost breakdown
    print("LLM Cost Breakdown:")
    total_cost = 0.0
    total_calls = 0
    
    for agent in all_agents:
        if hasattr(agent, 'get_cost_summary'):
            cost_info = agent.get_cost_summary()
            total_cost += cost_info['estimated_cost_usd']
            total_calls += cost_info['total_calls']
    
    if total_calls > 0:
        print(f"  Total LLM Calls:        {total_calls:,}")
        print(f"  Total Cost:             ${total_cost:.2f}")
        print(f"  Cost per Call:          ${total_cost/total_calls:.4f}")
        print(f"  Cost per Agent:         ${total_cost/len(all_agents):.4f}")
    else:
        print("  No LLM calls made (using rule-based fallbacks)")
    print()
    
    # Metrics summary
    metrics_summary = results['metrics_summary']
    print("Metrics Summary:")
    print(f"  Average Actions/Step:   {metrics_summary['average_actions_per_step']:.1f}")
    print(f"  Average Step Duration:  {metrics_summary['average_step_duration_seconds']:.3f}s")
    print()


def main():
    """Run the complete simulation example."""
    
    # Print header
    print_simulation_header()
    
    # Configuration
    print("⚙️  Configuration")
    print("-" * 70)
    
    NUM_AGENTS = 100
    NUM_STEPS = 1000
    
    llm_config = {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "temperature": 0.7,
        "max_tokens": 256
    }
    
    sim_config = SimulationConfig(
        name="urban_simulation_100",
        start_time=datetime(2024, 1, 1, 8, 0, 0),  # 8 AM
        time_step_seconds=300,  # 5 minutes per step
        max_steps=NUM_STEPS,
        log_interval=100
    )
    
    print(f"  Number of Agents:       {NUM_AGENTS}")
    print(f"  Simulation Steps:       {NUM_STEPS}")
    print(f"  Time per Step:          {sim_config.time_step_seconds} seconds (5 minutes)")
    print(f"  Simulation Duration:    {NUM_STEPS * 5 / 60:.1f} hours")
    print(f"  LLM Model:              {llm_config['model']}")
    print()
    
    # Create simulation
    print("🏗️  Creating Simulation...")
    sim = Simulation("urban_simulation_100", sim_config.dict())
    
    # Set up environment
    print("🏙️  Setting Up Environment...")
    setup_urban_environment(sim)
    
    # Create agents
    print(f"👥 Creating {NUM_AGENTS} Diverse Agents...")
    agents = create_diverse_agents(NUM_AGENTS, llm_config)
    
    # Add agents to simulation
    print("📝 Registering Agents...")
    for agent in agents:
        sim.add_agent(agent, initial_location="home")
    
    print(f"✅ Setup Complete: {len(sim.agents)} agents ready")
    print()
    
    # Run simulation
    print("🚀 Starting Simulation...")
    print("=" * 70)
    
    try:
        # Run with periodic updates
        results = sim.run(num_steps=NUM_STEPS)
        
        # Show periodic updates every 200 steps
        # (This is handled by simulation's log_interval, but we can add more detail)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Simulation interrupted by user")
        results = sim._generate_results()
    
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Analyze results
    analyze_results(sim, results)
    
    # Save results
    print("💾 Saving Results...")
    csv_path = sim.save_results()
    print(f"  Metrics saved to: {csv_path}")
    
    # Save JSON version too
    json_path = sim.metrics.save_json()
    print(f"  JSON saved to:    {json_path}")
    print()
    
    # Generate plot if matplotlib available
    try:
        import matplotlib
        print("📊 Generating Plots...")
        plot_path = Path(csv_path).parent / f"{sim.name}_plot.png"
        sim.metrics.plot_metrics(str(plot_path))
        print(f"  Plot saved to:    {plot_path}")
    except ImportError:
        print("  (matplotlib not installed, skipping plot generation)")
    
    print()
    print("=" * 70)
    print("✅ Simulation Complete!")
    print("=" * 70)
    print()
    
    # Sample agent stories
    print("📖 Sample Agent Stories:")
    print("-" * 70)
    
    # Show detailed info for 3 random agents
    sample_agents = random.sample(list(sim.agents.values()), min(3, len(sim.agents)))
    
    for agent in sample_agents:
        print(f"\n{agent.profile.name} ({agent.profile.age}, {agent.profile.occupation})")
        print(f"  Personality: ", end="")
        for trait, value in list(agent.profile.personality_traits.items())[:3]:
            print(f"{trait}={value:.2f} ", end="")
        print()
        
        print(f"  Final Location:  {sim.environment.get_agent_location(agent.profile.agent_id)}")
        print(f"  Final Activity:  {agent.state.current_activity}")
        
        if hasattr(agent.state, 'needs'):
            print(f"  Final Needs:")
            for need, value in agent.state.needs.items():
                print(f"    {need:20s}: {value:.2f}")
        
        stats = agent.get_stats()
        print(f"  Actions Taken:   {stats['actions_taken']}")
        print(f"  Memory Size:     {stats['memory_size']}")
    
    print()


if __name__ == "__main__":
    # Check for API key
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  WARNING: OPENAI_API_KEY not found in environment")
        print("This example requires an OpenAI API key.")
        print("Set it in your .env file or export it:")
        print("  export OPENAI_API_KEY='sk-...'")
        print()
        response = input("Continue anyway? (agents will use rule-based fallback) [y/N]: ")
        if response.lower() != 'y':
            print("Exiting.")
            sys.exit(0)
        print()
    
    main()