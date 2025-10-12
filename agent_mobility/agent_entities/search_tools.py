"""Wrap the required methods from AgentMobilitySystem in tools"""

from agent_mobility import AgentMobilitySystem
from langchain import tools
from pydantic import BaseModel, Field

#helper to make lat,lng readable as address, used in stateModel
#TODO: change this to google maps api
from geopy.geocoders import Nominatim
def reverse_geocode(lat, lng):
    try:
        geolocator = Nominatim(user_agent="geoapi")
        location = geolocator.reverse((lat, lng), language="en")
        return location.address if location else (lat, lng)
    except:
        print("Reverse geocoding did not work")
        return (lat, lng)

#entity's state
class stateModel(BaseModel):
    current_location: str = Field(..., "This is where you are right now")
    address: str = Field(..., "This is where you live")
    intends_to_go_somewhere: bool = Field(..., )
    destination: str = Field(..., "This is where you are headed right now")
    
@tool
def get_state(system: AgentMobilitySystem, entity_id: str) -> stateModel:

    state = system.get_entity_state()
    current_location = reverse_geocode(state.current_location.lat, state.current_location.lng)

    intends_to_go_somewhere = False
    if state.destination != None:
        intends_to_go_somewhere = True
    
    