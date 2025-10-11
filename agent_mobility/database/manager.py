"""Database manager for agent mobility system"""

import sqlite3
from contextlib import contextmanager
from typing import List, Dict, Optional


class DatabaseManager:
    """Manages SQLite database operations"""
    
    def __init__(self, db_path: str = 'navigation.db'):
        """Initialize database connection and create tables"""
        self.db_path = db_path
        self.init_database()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def init_database(self):
        """Create database tables if they don't exist"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Entities table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    entity_id TEXT PRIMARY KEY,
                    current_lat REAL NOT NULL,
                    current_lng REAL NOT NULL,
                    current_address TEXT,
                    destination_place_id TEXT,
                    destination_data TEXT,
                    last_updated TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Search history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS search_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    query TEXT NOT NULL,
                    radius INTEGER NOT NULL,
                    results_count INTEGER NOT NULL,
                    FOREIGN KEY (entity_id) REFERENCES entities (entity_id)
                )
            """)
            
            # Navigation history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS navigation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    place_id TEXT NOT NULL,
                    place_name TEXT NOT NULL,
                    place_address TEXT NOT NULL,
                    place_lat REAL NOT NULL,
                    place_lng REAL NOT NULL,
                    place_rating REAL,
                    place_data TEXT NOT NULL,
                    FOREIGN KEY (entity_id) REFERENCES entities (entity_id)
                )
            """)
            
            # Create indexes for better query performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_search_entity 
                ON search_history(entity_id, timestamp)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_nav_entity 
                ON navigation_history(entity_id, timestamp)
            """)
