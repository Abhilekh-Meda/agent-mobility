from agent_mobility import AgentMobilitySystem
from agent_mobility import Entity
import os
from dotenv import load_dotenv
from agent_mobility import TransportMode

load_dotenv()

google_maps_api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
db = "test.db"

sys = AgentMobilitySystem(google_maps_api_key, db)

# for i in range(1, 101):
#   sys.create_entity(i, i+32, i-117, "bruhville")

# for entity in sys.get_all_entities():
#   print (sys.get_entity(entity))

entity = Entity(sys, "1")
entity.update_location(32.7157, -117.1611 , "coolville")
print(entity)

print(entity.state)

results, status = entity.search_nearby("coffee shops", 5000 , 5)

for result in results:
  place = result
  duration = place.travel_times["driving"].duration_text
  distance = place.travel_times["driving"].distance_text
  print(place.name + " " + place.address + " " + " ------ " + duration + " " + distance)