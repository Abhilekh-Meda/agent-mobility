# Agent Mobility System - Complete Reference Guide
## This will evry likely change very soon so keep updating as needed

## Quick Start

```python
from agent_mobility import AgentMobilitySystem

# Initialize
system = AgentMobilitySystem(api_key="YOUR_GOOGLE_MAPS_API_KEY")

# Create an entity
entity = system.create_entity("agent_001", 32.7157, -117.1611, "San Diego")

# Search for places
results = system.search_nearby("agent_001", "coffee shops", radius=2000)

# Set a destination
system.set_destination("agent_001", results[0])
```

---

## DatabaseManager

**Purpose:** Manages SQLite database operations for the navigation system.

### Constructor
```python
DatabaseManager(db_path: str = 'navigation.db')
```
- Creates database connection
- Automatically calls `init_database()` to create tables

### Methods

#### `get_connection()`
- **Returns:** Context manager for database connections
- **Auto-commits** on success, **auto-rollbacks** on error, **auto-closes**
- **Usage:** `with db.get_connection() as conn: ...`

#### `init_database()`
- **Returns:** `None`
- Creates 3 tables if they don't exist:
  - `entities` - Entity locations and destinations
  - `search_history` - Past searches
  - `navigation_history` - Past destinations
- Creates indexes for performance
- Safe to call multiple times

---

## AgentMobilitySystem

**Purpose:** Main system for managing entity navigation and place searches.

### Constructor
```python
AgentMobilitySystem(api_key: str, db_path: str = 'navigation.db')
```
- `api_key`: Your Google Maps API key (requires Places API + Distance Matrix API)
- `db_path`: Path to SQLite database file
- Initializes Google Maps client and DatabaseManager

---

## Entity Management Methods

### `create_entity(entity_id, lat, lng, address=None)` 
**Returns:** `EntityState`

Creates a new entity at a location.

```python
entity = system.create_entity("agent_001", 32.7157, -117.1611, "San Diego")
# Returns EntityState with entity_id, location, destination, last_updated
```

### `get_entity_state(entity_id)`
**Returns:** `Optional[EntityState]` (None if not found)

Get current state of an entity.

```python
entity = system.get_entity_state("agent_001")
if entity:
    print(entity.current_location.lat)  # 32.7157
```

### `get_all_entities()`
**Returns:** `List[str]`

Get list of all entity IDs.

```python
entity_ids = system.get_all_entities()
# ["agent_001", "agent_002", "agent_003"]
```

### `update_location(entity_id, lat, lng, address=None)`
**Returns:** `None`

Update an entity's current location.

```python
system.update_location("agent_001", 32.7150, -117.1600, "Downtown SD")
```

### `delete_entity(entity_id)`
**Returns:** `None`

Delete an entity and all associated data.

```python
system.delete_entity("agent_001")
```

---

## Place Search & Navigation Methods

### `search_nearby(entity_id, query, radius=5000, max_results=20)`
**Returns:** `List[PlaceResult]`

Search for places near entity's current location.

```python
results = system.search_nearby(
    "agent_001", 
    "coffee shops", 
    radius=2000,      # meters
    max_results=10
)
# Returns list of PlaceResult objects with travel times
```

**Parameters:**
- `entity_id`: ID of searching entity
- `query`: Search term (e.g., "restaurants", "gas stations")
- `radius`: Search radius in meters (default: 5000)
- `max_results`: Max results to return (default: 20)

### `get_place_details(place_id)`
**Returns:** `Dict`

Get detailed information about a specific place.

```python
details = system.get_place_details("ChIJXXXXXXXX")
# Includes: name, address, rating, reviews, hours, phone, website, photos
```

### `set_destination(entity_id, place)`
**Returns:** `None`

Set a destination for an entity.

```python
place = results[0]
system.set_destination("agent_001", place)
# Updates entity's destination and saves to navigation_history
```

---

## History Retrieval Methods

### `get_search_history(entity_id, limit=50)`
**Returns:** `List[Dict]`

Get entity's past searches.

```python
history = system.get_search_history("agent_001", limit=10)
for search in history:
    print(search['timestamp'])      # "2025-10-11T14:30:00"
    print(search['query'])          # "coffee shops"
    print(search['radius'])         # 2000
    print(search['results_count'])  # 10
```

### `get_navigation_history(entity_id, limit=50)`
**Returns:** `List[Dict]`

Get entity's past destinations.

```python
history = system.get_navigation_history("agent_001", limit=10)
for nav in history:
    print(nav['timestamp'])      # "2025-10-11T14:30:00"
    print(nav['place_name'])     # "Starbucks"
    print(nav['place_address'])  # "123 Main St"
    print(nav['place_rating'])   # 4.5
    print(nav['place_lat'])      # 32.7150
    print(nav['place_lng'])      # -117.1600
```

---

## Entity (Wrapper Class)

**Purpose:** Convenient wrapper for entity-specific navigation operations. Eliminates need to repeatedly pass `entity_id` to system methods.

### Constructor
```python
entity = Entity(system, "agent_001")
# Or get from system:
entity = system.get_entity_state("agent_001")
```

### Properties

#### `state`
**Returns:** `Optional[EntityState]`

Get the current state of this entity.

```python
state = entity.state
# Access: entity_id, current_location, destination, last_updated
```

#### `current_location`
**Returns:** `Optional[Location]`

Get the current location.

```python
loc = entity.current_location
print(loc.lat, loc.lng)
```

#### `destination`
**Returns:** `Optional[PlaceResult]`

Get the current destination.

```python
dest = entity.destination
if dest:
    print(dest.name)
```

### Methods

#### `search_nearby(query, radius=5000, max_results=20, transport_mode=None)`
**Returns:** `List[PlaceResult]`

Search for places near this entity.

```python
results = entity.search_nearby("coffee shops", radius=2000)
# Same as: system.search_nearby(entity_id, "coffee shops", 2000)
```

**Parameters:**
- `query`: Search term
- `radius`: Search radius in meters
- `max_results`: Max results to return
- `transport_mode`: Optional specific transport mode

#### `set_destination(place)`
**Returns:** `None`

Set a destination for this entity.

```python
entity.set_destination(results[0])
# Same as: system.set_destination(entity_id, place)
```

#### `update_location(lat, lng, address=None)`
**Returns:** `None`

Update this entity's location.

```python
entity.update_location(32.7150, -117.1600, "Downtown")
# Same as: system.update_location(entity_id, lat, lng, address)
```

#### `get_search_history(limit=50)`
**Returns:** `List[Dict]`

Get this entity's search history.

```python
history = entity.get_search_history(10)
# Same as: system.get_search_history(entity_id, 10)
```

#### `get_navigation_history(limit=50)`
**Returns:** `List[Dict]`

Get this entity's navigation history.

```python
history = entity.get_navigation_history(10)
# Same as: system.get_navigation_history(entity_id, 10)
```

#### `delete()`
**Returns:** `None`

Delete this entity and all associated data.

```python
entity.delete()
# Same as: system.delete_entity(entity_id)
```

### String Representation

```python
print(entity)
# Output: Entity(id='agent_001', location=(32.7157, -117.1611))

repr(entity)
# Same output
```

### Example Usage

```python
# Instead of repeatedly passing entity_id to system:
# OLD WAY:
system.search_nearby(entity_id, "coffee", 2000)
system.set_destination(entity_id, place)
system.update_location(entity_id, 32.7150, -117.1600)

# NEW WAY (with Entity wrapper):
entity = Entity(system, entity_id)
entity.search_nearby("coffee", 2000)
entity.set_destination(place)
entity.update_location(32.7150, -117.1600)

# Or cleaner:
results = entity.search_nearby("restaurants", 3000)
if results:
    best = results[0]
    entity.set_destination(best)
    print(f"Navigating to {entity.destination.name}")
    print(f"Travel time: {best.travel_times['driving'].duration_text}")
```

---

## Data Models

### `EntityState`
Represents the complete state of an entity.

**Attributes:**
```python
entity.entity_id              # str: Unique identifier
entity.current_location       # Location: Current position
entity.destination            # Optional[PlaceResult]: Current destination
entity.last_updated           # str: ISO timestamp
```

**Location attributes:**
```python
entity.current_location.lat      # float: Latitude (-90 to 90)
entity.current_location.lng      # float: Longitude (-180 to 180)
entity.current_location.address  # Optional[str]: Address
```

---

### `PlaceResult`
Represents a place found in search results.

**Basic Attributes:**
```python
place.place_id              # str: Google Maps unique ID
place.name                  # str: Place name
place.address               # str: Place address
place.types                 # List[str]: Categories (e.g., ["cafe", "food"])
```

**Location:**
```python
place.location.lat          # float: Latitude
place.location.lng          # float: Longitude
place.location.address      # Optional[str]: Full address
```

**Ratings & Reviews:**
```python
place.rating                # Optional[float]: 0-5 rating
place.user_ratings_total    # Optional[int]: Number of reviews
```

**Price & Status:**
```python
place.price_level           # Optional[int]: 1-4 (1=cheap, 4=expensive)
place.open_now              # Optional[bool]: True/False/None
```

**Travel Times (Key Feature):**
```python
place.travel_times          # Dict: Travel info by mode

# Access by transport mode
place.travel_times['driving']
place.travel_times['walking']
place.travel_times['bicycling']
place.travel_times['transit']
```

**Convert to Dict:**
```python
place_dict = place.to_dict()  # Serializable dictionary
```

---

### `Location`
Represents a geographic location.

**Attributes:**
```python
location.lat                # float: Latitude
location.lng                # float: Longitude
location.address            # Optional[str]: Address string
```

**Format:**
- Latitude: -90.0 to 90.0 (negative = South)
- Longitude: -180.0 to 180.0 (negative = West)

**Examples:**
```python
Location(32.7157, -117.1611, "San Diego")      # San Diego
Location(40.7128, -74.0060, "New York")        # New York
Location(35.6762, 139.6503, "Tokyo")           # Tokyo
Location(-33.8688, 151.2093, "Sydney")         # Sydney
```

---

### `TravelInfo`
Represents travel details between two points.

**Attributes:**
```python
travel_info.mode               # str: "driving", "walking", "bicycling", "transit"
travel_info.duration_minutes   # int: Travel time in minutes
travel_info.distance_meters    # int: Distance in meters
travel_info.duration_text      # str: Human readable (e.g., "5 mins")
travel_info.distance_text      # str: Human readable (e.g., "1.2 km")
```

**Access via PlaceResult:**
```python
place.travel_times['driving'].duration_minutes      # 5
place.travel_times['driving'].distance_meters       # 1200
place.travel_times['driving'].duration_text         # "5 mins"
place.travel_times['walking'].duration_text         # "15 mins"
```

---

## Database Tables

### `entities`
Stores entity information.

| Column | Type | Purpose |
|--------|------|---------|
| entity_id | TEXT (PK) | Unique identifier |
| current_lat | REAL | Current latitude |
| current_lng | REAL | Current longitude |
| current_address | TEXT | Current address |
| destination_place_id | TEXT | Google Places ID of destination |
| destination_data | TEXT | JSON data of destination |
| last_updated | TEXT | ISO timestamp |
| created_at | TEXT | ISO timestamp |

### `search_history`
Tracks place searches.

| Column | Type | Purpose |
|--------|------|---------|
| id | INTEGER (PK) | Auto-increment |
| entity_id | TEXT (FK) | References entities |
| timestamp | TEXT | When search occurred |
| query | TEXT | Search term |
| radius | INTEGER | Search radius (meters) |
| results_count | INTEGER | Number of results |

### `navigation_history`
Tracks destinations set.

| Column | Type | Purpose |
|--------|------|---------|
| id | INTEGER (PK) | Auto-increment |
| entity_id | TEXT (FK) | References entities |
| timestamp | TEXT | When set as destination |
| place_id | TEXT | Google Places ID |
| place_name | TEXT | Place name |
| place_address | TEXT | Place address |
| place_lat | REAL | Latitude |
| place_lng | REAL | Longitude |
| place_rating | REAL | Rating (0-5) |
| place_data | TEXT | JSON of full place data |

---

## Common Use Cases

### Create Entity and Search
```python
system = AgentMobilitySystem(api_key="YOUR_KEY")
entity = system.create_entity("agent_001", 32.7157, -117.1611)
results = system.search_nearby("agent_001", "restaurants", 2000)
print(f"Found {len(results)} restaurants")
```

### Get Best Result by Travel Time
```python
results = system.search_nearby("agent_001", "coffee", 3000)
best = min(results, key=lambda p: p.travel_times['driving'].duration_minutes)
print(f"Closest: {best.name} - {best.travel_times['driving'].duration_text}")
```

### Set Destination and Check History
```python
system.set_destination("agent_001", results[0])
nav_history = system.get_navigation_history("agent_001", 5)
print(f"Visited {len(nav_history)} places")
```

### Track Entity Movements
```python
system.update_location("agent_001", 32.7150, -117.1600, "Downtown")
entity = system.get_entity_state("agent_001")
print(f"Entity at: {entity.current_location.lat}, {entity.current_location.lng}")
```

### Compare Transport Modes
```python
results = system.search_nearby("agent_001", "gym", 5000, 1)
place = results[0]
for mode in ['driving', 'walking', 'bicycling', 'transit']:
    time = place.travel_times[mode].duration_text
    distance = place.travel_times[mode].distance_text
    print(f"{mode.capitalize()}: {time} ({distance})")
```

---

## Error Handling

```python
try:
    entity = system.get_entity_state("nonexistent_agent")
    if not entity:
        print("Entity not found")
except ValueError as e:
    print(f"Error: {e}")

try:
    system.create_entity("agent_001", 0, 0)
except ValueError as e:
    print(f"Entity already exists")
```

---

## API Requirements

**Google Maps API Key needs:**
1. **Places API** - for searching and getting place details
2. **Distance Matrix API** - for calculating travel times

**Enable in Google Cloud Console:**
- Go to Console → Project → APIs & Services → Library
- Search and enable "Places API"
- Search and enable "Distance Matrix API"
- Create API key in Credentials section