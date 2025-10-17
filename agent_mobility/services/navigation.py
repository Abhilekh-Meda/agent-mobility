"""Core navigation system for AI agents"""

import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import googlemaps
from googlemaps.distance_matrix import distance_matrix
from googlemaps.places import places_nearby

from ..models.entities import Location, TravelInfo, PlaceResult, EntityState
from ..models.enums import TransportMode
from ..database.manager import DatabaseManager


class AgentMobilitySystem:
    """Main system for managing entity navigation"""
    
    def __init__(self, api_key: str, db_path: str = 'navigation.db'):
        """
        Initialize the navigation system
        
        Args:
            api_key: Google Maps API key
            db_path: Path to SQLite database file
        """
        self.gmaps = googlemaps.Client(key=api_key)
        self.db = DatabaseManager(db_path)
    
    def search_nearby(
        self,
        entity_id: str,
        query: str,
        radius: int = 5000,
        max_results: int = 20,
        transport_mode: Optional[TransportMode] = None
    ) -> Tuple[List[PlaceResult], str]:
        """
        Search for places near the entity's current location
        
        Args:
            entity_id: ID of the entity searching
            query: Search query (e.g., "restaurants", "coffee shops")
            radius: Search radius in meters
            max_results: Maximum number of results
            transport_mode: Optional specific transport mode to calculate travel times.
                        If None, calculates for all modes.
            
        Returns:
            Tuple of (List of PlaceResult objects with travel times, status string)
        """
        entity = self.get_entity_state(entity_id)
        if not entity:
            raise ValueError(f"Entity {entity_id} not found")
        
        origin = (entity.current_location.lat, entity.current_location.lng)
        
        # Search for places
        places_result = places_nearby(
            self.gmaps,
            location=origin,
            radius=radius,
            keyword=query
        )
        
        # Get status from API response
        status = places_result.get('status', 'UNKNOWN')
        
        results = []
        destinations = []
        
        # Process results
        for place in places_result.get('results', [])[:max_results]:
            loc = place['geometry']['location']
            place_result = PlaceResult(
                place_id=place['place_id'],
                name=place['name'],
                address=place.get('vicinity', ''),
                location=Location(lat=loc['lat'], lng=loc['lng']),
                rating=place.get('rating'),
                user_ratings_total=place.get('user_ratings_total'),
                types=place.get('types', []),
                price_level=place.get('price_level'),
                open_now=place.get('opening_hours', {}).get('open_now'),
                travel_times={}
            )
            results.append(place_result)
            destinations.append((loc['lat'], loc['lng']))
        
        # Get travel times for specified transport mode(s)
        if destinations:
            if transport_mode:
                # Calculate for single specified mode
                travel_times = self._get_travel_times(
                    origin,
                    destinations,
                    transport_mode.value
                )
                
                for i, result in enumerate(results):
                    if i < len(travel_times):
                        result.travel_times[transport_mode.value] = travel_times[i]
            else:
                # Calculate for all transport modes (original behavior)
                for mode in TransportMode:
                    travel_times = self._get_travel_times(
                        origin,
                        destinations,
                        mode.value
                    )
                    
                    for i, result in enumerate(results):
                        if i < len(travel_times):
                            result.travel_times[mode.value] = travel_times[i]
        
        # Save search to database
        self._save_search_history(entity_id, query, radius, len(results))
        
        return results, status
    
    def _get_travel_times(
        self,
        origin: tuple,
        destinations: List[tuple],
        mode: str
    ) -> List[TravelInfo]:
        """
        Get travel times from origin to multiple destinations
        
        Args:
            origin: (lat, lng) tuple
            destinations: List of (lat, lng) tuples
            mode: Transport mode
            
        Returns:
            List of TravelInfo objects
        """
        result = distance_matrix(
            self.gmaps,
            origins=[origin],
            destinations=destinations,
            mode=mode,
            units='metric'
        )
        
        travel_infos = []
        
        for element in result['rows'][0]['elements']:
            if element['status'] == 'OK':
                travel_infos.append(TravelInfo(
                    mode=mode,
                    duration_minutes=element['duration']['value'] // 60,
                    distance_meters=element['distance']['value'],
                    duration_text=element['duration']['text'],
                    distance_text=element['distance']['text']
                ))
            else:
                # Handle unavailable routes
                travel_infos.append(TravelInfo(
                    mode=mode,
                    duration_minutes=0,
                    distance_meters=0,
                    duration_text="N/A",
                    distance_text="N/A"
                ))
        
        return travel_infos
    
    def get_place_details(self, place_id: str) -> Tuple[Dict, str]:
        """
        Get detailed information about a specific place
        
        Args:
            place_id: Google Maps place ID
            
        Returns:
            Tuple of (Dictionary with place details including reviews, status string)
        """
        result = self.gmaps.place(
            place_id=place_id,
            fields=[
                'name', 'formatted_address', 'rating', 'reviews',
                'user_ratings_total', 'price_level', 'opening_hours',
                'formatted_phone_number', 'website', 'photos'
            ]
        )
        
        status = result.get('status', 'UNKNOWN')
        return result.get('result', {}), status
    
    def set_destination(self, entity_id: str, place: PlaceResult):
        """
        Set a destination for an entity
        
        Args:
            entity_id: ID of the entity
            place: PlaceResult to set as destination
        """
        entity = self.get_entity_state(entity_id)
        if not entity:
            raise ValueError(f"Entity {entity_id} not found")
        
        timestamp = datetime.now().isoformat()
        place_data = json.dumps(place.to_dict())
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Update entity's destination
            cursor.execute("""
                UPDATE entities 
                SET destination_place_id = ?,
                    destination_data = ?,
                    last_updated = ?
                WHERE entity_id = ?
            """, (place.place_id, place_data, timestamp, entity_id))
            
            # Add to navigation history
            cursor.execute("""
                INSERT INTO navigation_history 
                (entity_id, timestamp, place_id, place_name, place_address,
                 place_lat, place_lng, place_rating, place_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entity_id, timestamp, place.place_id, place.name, place.address,
                place.location.lat, place.location.lng, place.rating, place_data
            ))
    
    def update_location(self, entity_id: str, lat: float, lng: float, address: str = None):
        """
        Update an entity's current location
        
        Args:
            entity_id: ID of the entity
            lat: Latitude
            lng: Longitude
            address: Optional address string
        """
        entity = self.get_entity_state(entity_id)
        if not entity:
            raise ValueError(f"Entity {entity_id} not found")
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE entities 
                SET current_lat = ?,
                    current_lng = ?,
                    current_address = ?,
                    last_updated = ?
                WHERE entity_id = ?
            """, (lat, lng, address, datetime.now().isoformat(), entity_id))
    
    def create_entity(self, entity_id: str, lat: float, lng: float, address: str = None) -> EntityState:
        """
        Create a new entity in the system
        
        Args:
            entity_id: Unique ID for the entity
            lat: Initial latitude
            lng: Initial longitude
            address: Optional address string
            
        Returns:
            Created EntityState
        """
        now = datetime.now().isoformat()
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Check if entity already exists
            cursor.execute("SELECT entity_id FROM entities WHERE entity_id = ?", (entity_id,))
            if cursor.fetchone():
                raise ValueError(f"Entity {entity_id} already exists")
            
            # Create new entity
            cursor.execute("""
                INSERT INTO entities 
                (entity_id, current_lat, current_lng, current_address, 
                 last_updated, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (entity_id, lat, lng, address, now, now))
        
        return EntityState(
            entity_id=entity_id,
            current_location=Location(lat=lat, lng=lng, address=address),
            destination=None,
            last_updated=now
        )
    
    def get_entity_state(self, entity_id: str) -> Optional[EntityState]:
        """Get the current state of an entity"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT entity_id, current_lat, current_lng, current_address,
                       destination_data, last_updated
                FROM entities
                WHERE entity_id = ?
            """, (entity_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            destination = None
            if row['destination_data']:
                dest_dict = json.loads(row['destination_data'])
                destination = PlaceResult(
                    place_id=dest_dict['place_id'],
                    name=dest_dict['name'],
                    address=dest_dict['address'],
                    location=Location(**dest_dict['location']),
                    rating=dest_dict['rating'],
                    user_ratings_total=dest_dict['user_ratings_total'],
                    types=dest_dict['types'],
                    price_level=dest_dict['price_level'],
                    open_now=dest_dict['open_now'],
                    travel_times=dest_dict['travel_times']
                )
            
            return EntityState(
                entity_id=row['entity_id'],
                current_location=Location(
                    lat=row['current_lat'],
                    lng=row['current_lng'],
                    address=row['current_address']
                ),
                destination=destination,
                last_updated=row['last_updated']
            )
    
    def get_search_history(self, entity_id: str, limit: int = 50) -> List[Dict]:
        """Get search history for an entity"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT timestamp, query, radius, results_count
                FROM search_history
                WHERE entity_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (entity_id, limit))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_navigation_history(self, entity_id: str, limit: int = 50) -> List[Dict]:
        """Get navigation history for an entity"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT timestamp, place_id, place_name, place_address,
                       place_lat, place_lng, place_rating
                FROM navigation_history
                WHERE entity_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (entity_id, limit))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def _save_search_history(self, entity_id: str, query: str, radius: int, results_count: int):
        """Save a search to history"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO search_history (entity_id, timestamp, query, radius, results_count)
                VALUES (?, ?, ?, ?, ?)
            """, (entity_id, datetime.now().isoformat(), query, radius, results_count))
            
            # Also update entity's last_updated timestamp
            cursor.execute("""
                UPDATE entities SET last_updated = ? WHERE entity_id = ?
            """, (datetime.now().isoformat(), entity_id))
    
    def delete_entity(self, entity_id: str):
        """Delete an entity and all associated data"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM search_history WHERE entity_id = ?", (entity_id,))
            cursor.execute("DELETE FROM navigation_history WHERE entity_id = ?", (entity_id,))
            cursor.execute("DELETE FROM entities WHERE entity_id = ?", (entity_id,))
    
    def get_all_entities(self) -> List[str]:
        """Get list of all entity IDs"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT entity_id FROM entities ORDER BY created_at")
            return [row['entity_id'] for row in cursor.fetchall()]

    def get_entity(self, entity_id: str) -> 'Entity':
        """
        Get an Entity wrapper instance for convenient method access
        
        Args:
            entity_id: ID of the entity
            
        Returns:
            Entity instance that wraps this system
            
        Raises:
            ValueError: If entity doesn't exist
        """
        from .entity import Entity
        
        # Verify entity exists
        if not self.get_entity_state(entity_id):
            raise ValueError(f"Entity {entity_id} not found")
        
        return Entity(self, entity_id)