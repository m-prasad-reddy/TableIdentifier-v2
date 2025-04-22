import os
import json
from typing import Dict, Any
import logging
import logging.config
import pyodbc
import time

class SchemaManager:
    """Manages database schema information, including tables, views, and relationships.

    Provides methods to build, cache, and validate schema data from a database.
    """

    def __init__(self, db_name: str):
        """Initialize the schema manager.

        Args:
            db_name (str): Name of the database.
        """
        # Ensure logs directory exists
        os.makedirs("logs", exist_ok=True)
        logging_config_path = "app-config/logging_config.ini"
        if os.path.exists(logging_config_path):
            try:
                logging.config.fileConfig(logging_config_path, disable_existing_loggers=False)
            except Exception as e:
                print(f"Error loading logging config: {e}")
        
        self.logger = logging.getLogger("schema")
        self.db_name = db_name
        self.cache_path = os.path.join("schema_cache", db_name, "schema.json")
        self.logger.debug(f"Initialized SchemaManager for {db_name}")

    def needs_refresh(self, connection: pyodbc.Connection) -> bool:
        """Check if the schema cache needs to be refreshed.

        Args:
            connection (pyodbc.Connection): Database connection.

        Returns:
            bool: True if refresh is needed, False otherwise.
        """
        if not os.path.exists(self.cache_path):
            self.logger.debug("Schema cache does not exist, needs refresh")
            return True
        
        try:
            with connection.cursor() as cursor:
                cursor.execute("""
                    SELECT MAX(last_user_update) 
                    FROM sys.dm_db_index_usage_stats 
                    WHERE database_id = DB_ID()
                """)
                last_update = cursor.fetchone()[0]
                
                if last_update is None:
                    self.logger.debug("No index usage stats, checking file age")
                    return os.path.getmtime(self.cache_path) < (time.time() - 24*60*60)
                
                cache_time = os.path.getmtime(self.cache_path)
                self.logger.debug(f"Last update: {last_update}, Cache time: {cache_time}")
                return cache_time < last_update.timestamp()
        except Exception as e:
            self.logger.error(f"Error checking schema refresh: {e}")
            return True

    def build_data_dict(self, connection: pyodbc.Connection) -> Dict[str, Any]:
        """Build a schema dictionary from the database.

        Includes tables, views, columns, relationships, and validates consistency.

        Args:
            connection (pyodbc.Connection): Database connection.

        Returns:
            Dict[str, Any]: Schema dictionary with tables, views, columns, and relationships.

        Raises:
            Exception: If schema building fails.
        """
        schema_dict = {
            'tables': {},
            'views': {},
            'columns': {},
            'relationships': []
        }
        
        try:
            with connection.cursor() as cursor:
                # Get schemas
                cursor.execute("SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA")
                schemas = [row[0] for row in cursor.fetchall()]
                
                for schema in schemas:
                    schema_dict['tables'][schema] = []
                    schema_dict['views'][schema] = []
                    schema_dict['columns'][schema] = {}
                    
                    # Get tables
                    cursor.execute(f"""
                        SELECT TABLE_NAME 
                        FROM INFORMATION_SCHEMA.TABLES 
                        WHERE TABLE_SCHEMA = ? AND TABLE_TYPE = 'BASE TABLE'
                    """, schema)
                    tables = [row[0] for row in cursor.fetchall()]
                    schema_dict['tables'][schema] = tables
                    
                    # Get views
                    cursor.execute(f"""
                        SELECT TABLE_NAME 
                        FROM INFORMATION_SCHEMA.TABLES 
                        WHERE TABLE_SCHEMA = ? AND TABLE_TYPE = 'VIEW'
                    """, schema)
                    views = [row[0] for row in cursor.fetchall()]
                    schema_dict['views'][schema] = views
                    
                    # Get columns for tables and views
                    for table in tables + views:
                        cursor.execute(f"""
                            SELECT 
                                COLUMN_NAME,
                                DATA_TYPE,
                                IS_NULLABLE,
                                COLUMNPROPERTY(OBJECT_ID(? + '.' + ?), COLUMN_NAME, 'IsIdentity') AS is_identity,
                                (SELECT 1 
                                 FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
                                 JOIN INFORMATION_SCHEMA.CONSTRAINT_COLUMN_USAGE ccu
                                 ON tc.CONSTRAINT_NAME = ccu.CONSTRAINT_NAME
                                 WHERE tc.TABLE_SCHEMA = ? 
                                 AND tc.TABLE_NAME = ?
                                 AND tc.CONSTRAINT_TYPE = 'PRIMARY KEY'
                                 AND ccu.COLUMN_NAME = COLUMN_NAME) AS is_primary_key
                            FROM INFORMATION_SCHEMA.COLUMNS
                            WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
                        """, (schema, table, schema, table, schema, table))
                        
                        columns = {}
                        for row in cursor.fetchall():
                            columns[row[0]] = {
                                'type': row[1],
                                'nullable': row[2] == 'YES',
                                'is_identity': bool(row[3]),
                                'is_primary_key': bool(row[4])
                            }
                        schema_dict['columns'][schema][table] = columns
                
                # Get relationships
                cursor.execute("""
                    SELECT 
                        fk.TABLE_SCHEMA AS parent_schema,
                        fk.TABLE_NAME AS parent_table,
                        ccu.COLUMN_NAME AS parent_column,
                        pk.TABLE_SCHEMA AS child_schema,
                        pk.TABLE_NAME AS child_table,
                        ccu2.COLUMN_NAME AS child_column
                    FROM INFORMATION_SCHEMA.REFERENTIAL_CONSTRAINTS rc
                    JOIN INFORMATION_SCHEMA.TABLE_CONSTRAINTS fk
                        ON rc.CONSTRAINT_NAME = fk.CONSTRAINT_NAME
                    JOIN INFORMATION_SCHEMA.TABLE_CONSTRAINTS pk
                        ON rc.UNIQUE_CONSTRAINT_NAME = pk.CONSTRAINT_NAME
                    JOIN INFORMATION_SCHEMA.CONSTRAINT_COLUMN_USAGE ccu
                        ON fk.CONSTRAINT_NAME = ccu.CONSTRAINT_NAME
                    JOIN INFORMATION_SCHEMA.CONSTRAINT_COLUMN_USAGE ccu2
                        ON pk.CONSTRAINT_NAME = ccu2.CONSTRAINT_NAME
                """)
                
                for row in cursor.fetchall():
                    schema_dict['relationships'].append({
                        'parent': f"{row[0]}.{row[1]}.{row[2]}",
                        'child': f"{row[3]}.{row[4]}.{row[5]}"
                    })
                
                # Validate schema
                self._validate_schema(schema_dict)
                
                self._save_to_cache(schema_dict)
                self.logger.debug("Built and cached schema dictionary")
                return schema_dict
                
        except Exception as e:
            self.logger.error(f"Error building schema dictionary: {e}")
            raise

    def _validate_schema(self, schema_dict: Dict[str, Any]):
        """Validate the schema dictionary for consistency.

        Args:
            schema_dict (Dict[str, Any]): Schema dictionary to validate.

        Raises:
            ValueError: If the schema is invalid.
        """
        for schema in schema_dict['tables']:
            if schema not in schema_dict['columns']:
                self.logger.error(f"Missing columns for schema: {schema}")
                raise ValueError(f"Missing columns for schema: {schema}")
            
            for table in schema_dict['tables'][schema]:
                if table not in schema_dict['columns'][schema]:
                    self.logger.error(f"Missing columns for table: {schema}.{table}")
                    raise ValueError(f"Missing columns for table: {schema}.{table}")
            
            for view in schema_dict['views'][schema]:
                if view not in schema_dict['columns'][schema]:
                    self.logger.error(f"Missing columns for view: {schema}.{view}")
                    raise ValueError(f"Missing columns for view: {schema}.{view}")
        
        for rel in schema_dict['relationships']:
            parent_parts = rel['parent'].split('.')
            child_parts = rel['child'].split('.')
            if len(parent_parts) != 3 or len(child_parts) != 3:
                self.logger.error(f"Invalid relationship format: {rel}")
                raise ValueError(f"Invalid relationship format: {rel}")
            
            parent_schema, parent_table, _ = parent_parts
            child_schema, child_table, _ = child_parts
            
            if (parent_schema not in schema_dict['tables'] or
                parent_table not in schema_dict['tables'][parent_schema] or
                child_schema not in schema_dict['tables'] or
                child_table not in schema_dict['tables'][child_schema]):
                self.logger.error(f"Invalid relationship tables: {rel}")
                raise ValueError(f"Invalid relationship tables: {rel}")

        self.logger.debug("Schema validation successful")

    def load_from_cache(self) -> Dict[str, Any]:
        """Load schema dictionary from cache.

        Returns:
            Dict[str, Any]: Cached schema dictionary.

        Raises:
            Exception: If loading fails.
        """
        try:
            with open(self.cache_path, 'r') as f:
                schema_dict = json.load(f)
                self._validate_schema(schema_dict)
                self.logger.debug(f"Loaded schema from cache: {self.cache_path}")
                return schema_dict
        except Exception as e:
            self.logger.error(f"Error loading schema from cache: {e}")
            raise

    def _save_to_cache(self, schema_dict: Dict[str, Any]):
        """Save schema dictionary to cache.

        Args:
            schema_dict (Dict[str, Any]): Schema dictionary to save.

        Raises:
            Exception: If saving fails.
        """
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        try:
            with open(self.cache_path, 'w') as f:
                json.dump(schema_dict, f, indent=2)
            self.logger.debug(f"Saved schema to cache: {self.cache_path}")
        except Exception as e:
            self.logger.error(f"Error saving schema to cache: {e}")
            raise