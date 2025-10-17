# Agent Mobility System - Quick Reference
## This will keep changing with time, update as needed.

## Data Classes

### `Location`
Represents a geographic location
- `lat` - Latitude coordinate
- `lng` - Longitude coordinate
- `address` - Optional address string

### `TravelInfo`
Travel details for a transport mode
- `mode` - Transport mode (driving, walking, etc.)
- `duration_minutes` - Travel time in minutes
- `distance_meters` - Distance in meters
- `duration_text` - Human-readable duration (e.g., "15 mins")
- `distance_text` - Human-readable distance (e.g., "2.3 km")

### `PlaceResult`
A place from search results
- `place_id` - Google Maps place ID
- `name` - Place name
- `address` - Street address
- `location` - Location object with coordinates
- `rating` - Average rating (1-5)
- `user_ratings_total` - Number of ratings
- `types` - List of place types (e.g., ["restaurant", "food"])
- `price_level` - Price level (0-4, where 4 is most expensive)
- `open_now` - Whether currently open
- `travel_times` - Dict of transport mode → TravelInfo
- **Methods:**
  - `to_dict()` - Convert to dictionary

### `EntityState`
Current state of an entity
- `entity_id` - Unique entity identifier
- `current_location` - Current Location
- `destination` - Current PlaceResult destination (or None)
- `last_updated` - ISO timestamp of last update
- **Methods:**
  - `to_dict()` - Convert to dictionary

## Enums

### `TransportMode`
Available transport modes:
- `DRIVING` - Car/vehicle
- `WALKING` - On foot
- `BICYCLING` - By bicycle
- `TRANSIT` - Public transportation

## Main Classes

### `AgentMobilitySystem`
Main navigation system

**Initialization:**
- `__init__(api_key, db_path='navigation.db')` - Create system with Google Maps API key

**Entity Management:**
- `create_entity(entity_id, lat, lng, address=None)` → EntityState - Create new entity
- `get_entity_state(entity_id)` → EntityState | None - Get entity's current state
- `get_entity(entity_id)` → Entity - Get Entity wrapper instance
- `get_all_entities()` → List[str] - Get all entity IDs
- `delete_entity(entity_id)` - Delete entity and all its data
- `update_location(entity_id, lat, lng, address=None)` - Update entity's location

**Search & Navigation:**
- `search_nearby(entity_id, query, radius=5000, max_results=20, transport_mode=None)` → (List[PlaceResult], status) - Search nearby places
- `get_place_details(place_id)` → (Dict, status) - Get detailed place information
- `set_destination(entity_id, place)` - Set entity's destination

**History:**
- `get_search_history(entity_id, limit=50)` → List[Dict] - Get search history
- `get_navigation_history(entity_id, limit=50)` → List[Dict] - Get navigation history

### `Entity`
Wrapper for entity-specific operations

**Initialization:**
- `__init__(system, entity_id)` - Create wrapper (usually via `system.get_entity()`)

**Properties:**
- `state` → EntityState | None - Current entity state
- `current_location` → Location | None - Current location
- `destination` → PlaceResult | None - Current destination

**Methods:**
- `search_nearby(query, radius=5000, max_results=20, transport_mode=None)` → (List[PlaceResult], status) - Search nearby
- `get_place_details(place_id)` → (Dict, status) - Get place details
- `set_destination(place)` - Set destination
- `update_location(lat, lng, address=None)` - Update location
- `get_search_history(limit=50)` → List[Dict] - Get search history
- `get_navigation_history(limit=50)` → List[Dict] - Get navigation history
- `delete()` - Delete this entity

## Accessing Travel Times

The `travel_times` field in `PlaceResult` is a dictionary mapping transport mode strings to `TravelInfo` objects.

**Structure:**
```python
place.travel_times = {
    "driving": TravelInfo(...),
    "walking": TravelInfo(...),
    "bicycling": TravelInfo(...),
    "transit": TravelInfo(...)
}
```

**Accessing travel times:**
```python
# Get a specific mode
driving_info = place.travel_times["driving"]
walking_info = place.travel_times["walking"]

# Access TravelInfo fields
duration = driving_info.duration_minutes      # e.g., 15
distance = driving_info.distance_meters       # e.g., 2300
duration_text = driving_info.duration_text    # e.g., "15 mins"
distance_text = driving_info.distance_text    # e.g., "2.3 km"
mode = driving_info.mode                      # e.g., "driving"

# Check if a mode exists
if "transit" in place.travel_times:
    transit_time = place.travel_times["transit"].duration_minutes

# Loop through all modes
for mode, travel_info in place.travel_times.items():
    print(f"{mode}: {travel_info.duration_text}")
```

## Common Usage Patterns

```python
# Initialize system
system = AgentMobilitySystem(api_key="YOUR_API_KEY")

# Create entity
system.create_entity("agent1", lat=40.7128, lng=-74.0060)

# Get entity wrapper
entity = system.get_entity("agent1")

# Search nearby
results, status = entity.search_nearby("coffee shops", radius=1000)

# Access travel times
if results and status == "OK":
    first_place = results[0]
    
    # Get walking time
    walk_time = first_place.travel_times["walking"].duration_minutes
    print(f"Walk to {first_place.name}: {walk_time} mins")
    
    # Compare modes
    for mode in ["driving", "walking", "bicycling"]:
        info = first_place.travel_times[mode]
        print(f"{mode}: {info.duration_text} ({info.distance_text})")
    
    # Set destination
    entity.set_destination(first_place)

# Update location
entity.update_location(40.7150, -74.0070)

# Search with specific transport mode only
results, status = entity.search_nearby(
    "restaurants",
    transport_mode=TransportMode.WALKING
)
# Now results only have "walking" in travel_times
```

## Status Codes

API methods return status strings:
- `"OK"` - Success
- `"ZERO_RESULTS"` - No results found
- `"INVALID_REQUEST"` - Invalid parameters
- `"OVER_QUERY_LIMIT"` - API quota exceeded
- `"REQUEST_DENIED"` - API key issue
- `"UNKNOWN_ERROR"` - Server error