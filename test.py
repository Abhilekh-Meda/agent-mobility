from agent_mobility import AgentMobilitySystem
import os
import load_dotenv

load_dotenv()

sys = AgentMobilitySystem(os.environ.get("GOOGLE_MAPS_API_KEY"))

sys.create_entity("123", 32, -117)
sys.