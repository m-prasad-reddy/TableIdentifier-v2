import logging
import logging.config
import os
import json
from typing import Dict, List, Tuple
from database.connection import DatabaseConnection
from config.config_manager import DBConfigManager
from config.patterns import PatternManager
from schema.schema_manager import SchemaManager
from feedback.feedback_manager import FeedbackManager
from analysis.table_identifier import TableIdentifier
from analysis.name_match_manager import NameMatchManager
from analysis.processor import NLPPipeline
from nlp.query_processor import QueryProcessor
from cli.interface import DatabaseAnalyzerCLI

class DatabaseAnalyzer:
    """Orchestrates database schema analysis and natural language query processing.

    This class integrates components for database connection, schema management,
    feedback handling, and NLP-based query processing to enable natural language
    interaction with database schemas.
    """
    
    def __init__(self):
        """Initialize logging and core components with null states."""
        # Ensure log directory exists
        os.makedirs("logs", exist_ok=True)
        os.makedirs("schema_cache", exist_ok=True)
        os.makedirs("feedback_cache", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        
        # Set up logging from configuration file
        logging_config_path = "app-config/logging_config.ini"
        try:
            if os.path.exists(logging_config_path):
                logging.config.fileConfig(logging_config_path, disable_existing_loggers=False)
            else:
                logging.basicConfig(
                    level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler('logs/app.log'),
                        logging.StreamHandler()
                    ]
                )
                print(f"Warning: {logging_config_path} not found, using default logging")
        except Exception as e:
            logging.basicConfig(
                level=logging.DEBUG,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler('logs/app.log'),
                    logging.StreamHandler()
                ]
            )
            print(f"Error loading logging config: {e}")
        
        self.logger = logging.getLogger("analyzer")
        self.connection_manager = None
        self.config_manager = None
        self.schema_manager = None
        self.pattern_manager = None
        self.feedback_manager = None
        self.nlp_pipeline = None
        self.name_matcher = None
        self.table_identifier = None
        self.query_processor = None
        self.current_config = None
        self.schema_dict = {}
        self.query_history = []  # Track recent queries
        self.logger.debug("Initialized DatabaseAnalyzer")

    def run(self):
        """Launch the command-line interface and manage application shutdown."""
        try:
            cli = DatabaseAnalyzerCLI(self)
            cli.run()
        except Exception as e:
            self.logger.error(f"Application error: {e}")
            print(f"Application error: {e}")
        finally:
            # Save model and close connection
            if self.table_identifier and self.current_config:
                self.table_identifier.save_name_matches()
                self.table_identifier.save_model(f"models/{self.current_config['database']}_model.json")
            if self.connection_manager:
                self.connection_manager.close()
            self.logger.info("Application shutdown")

    def load_configs(self, config_path: str = "app-config/database_configurations.json") -> Dict:
        """Load and validate database configurations from a JSON file.

        Args:
            config_path (str): Path to the configuration file.

        Returns:
            Dict: Dictionary of valid database configurations.
        """
        if not os.path.exists(config_path):
            self.logger.warning(f"Config file not found at {config_path}")
            config_path = input("Enter config file path: ").strip()
            if not os.path.exists(config_path):
                self.logger.error(f"Invalid config path: {config_path}")
                return {}
        
        try:
            self.config_manager = DBConfigManager()
            configs = self.config_manager.load_configs(config_path)
            valid_configs = self._validate_configs(configs)
            self.logger.debug(f"Loaded {len(valid_configs)} valid configurations")
            return valid_configs
        except Exception as e:
            self.logger.error(f"Error loading configs: {e}")
            return {}

    def _validate_configs(self, configs: Dict) -> Dict:
        """Validate database configurations for required keys.

        Args:
            configs (Dict): Dictionary of configurations.

        Returns:
            Dict: Dictionary of valid configurations.
        """
        required_keys = ['database', 'server', 'driver']
        valid_configs = {}
        for name, config in configs.items():
            if all(key in config for key in required_keys):
                valid_configs[name] = config
            else:
                self.logger.warning(f"Invalid config {name}: missing keys {required_keys}")
        return valid_configs

    def set_current_config(self, config: Dict):
        """Set the active database configuration.

        Args:
            config (Dict): Configuration dictionary for the database.
        """
        self.current_config = config
        self.logger.debug(f"Set config: {config.get('database')}")

    def connect_to_database(self) -> bool:
        """Establish connection to the selected database and initialize components.

        Returns:
            bool: True if connection and initialization succeed, False otherwise.
        """
        if not self.current_config:
            self.logger.error("No configuration selected")
            print("No configuration selected")
            return False
        
        try:
            self.connection_manager = DatabaseConnection()
            if self.connection_manager.connect(self.current_config):
                self._initialize_managers()
                self.logger.info(f"Connected to {self.current_config['database']}")
                return True
            else:
                self.logger.error("Database connection failed")
                print("Database connection failed")
                return False
        except Exception as e:
            self.logger.error(f"Connection or initialization error: {e}")
            print(f"Connection or initialization error: {e}")
            self._reset_managers()
            return False

    def _initialize_managers(self):
        """Initialize component managers for schema and query processing."""
        db_name = self.current_config['database']
        self.logger.debug(f"Initializing managers for {db_name}")
        
        # Initialize schema manager
        self.schema_manager = SchemaManager(db_name)
        try:
            if self.schema_manager.needs_refresh(self.connection_manager.connection):
                self.logger.debug("Building fresh schema")
                self.schema_dict = self.schema_manager.build_data_dict(
                    self.connection_manager.connection
                )
            else:
                self.logger.debug("Loading schema from cache")
                self.schema_dict = self.schema_manager.load_from_cache()
        except Exception as e:
            self.logger.error(f"Schema initialization failed: {e}")
            raise
        
        # Initialize other managers
        self.pattern_manager = PatternManager(self.schema_dict)
        self.feedback_manager = FeedbackManager(db_name)
        self.nlp_pipeline = NLPPipeline(self.pattern_manager, db_name)
        self.name_matcher = NameMatchManager(db_name)
        
        # Initialize table identifier
        try:
            self.table_identifier = TableIdentifier(
                self.schema_dict,
                self.feedback_manager,
                self.pattern_manager,
                self.name_matcher,
                db_name
            )
        except Exception as e:
            self.logger.warning(f"TableIdentifier initialization failed, likely due to CSV parsing: {e}")
            self.table_identifier = None  # Allow partial initialization
        
        # Initialize query processor
        try:
            self.query_processor = QueryProcessor(self.table_identifier)
        except Exception as e:
            self.logger.error(f"QueryProcessor initialization failed: {e}")
            raise
        
        self.logger.debug("Managers initialized successfully")

    def _reset_managers(self):
        """Reset managers to null states on initialization failure."""
        self.schema_manager = None
        self.pattern_manager = None
        self.feedback_manager = None
        self.nlp_pipeline = None
        self.name_matcher = None
        self.table_identifier = None
        self.query_processor = None
        self.schema_dict = {}
        self.logger.debug("Managers reset due to initialization failure")

    def reload_all_configurations(self) -> bool:
        """Reload configurations and reinitialize managers.

        Returns:
            bool: True if reload is successful, False otherwise.
        """
        if not self.connection_manager or not self.connection_manager.is_connected():
            self.logger.error("Not connected to database")
            print("Not connected to database!")
            return False
        
        try:
            db_name = self.current_config['database']
            self.logger.debug(f"Reloading configurations for {db_name}")
            self.schema_dict = self.schema_manager.build_data_dict(
                self.connection_manager.connection
            )
            self.pattern_manager = PatternManager(self.schema_dict)
            self.feedback_manager = FeedbackManager(db_name)
            self.nlp_pipeline = NLPPipeline(self.pattern_manager, db_name)
            self.name_matcher = NameMatchManager(db_name)
            self.table_identifier = TableIdentifier(
                self.schema_dict,
                self.feedback_manager,
                self.pattern_manager,
                self.name_matcher,
                db_name
            )
            self.query_processor = QueryProcessor(self.table_identifier)
            self.logger.info("Configurations reloaded successfully")
            return True
        except Exception as e:
            self.logger.error(f"Reload failed: {e}")
            print(f"Reload failed: {e}")
            self._reset_managers()
            return False

    def process_query(self, query: str) -> Tuple[List[str], float]:
        """Process a natural language query to identify relevant tables.

        Args:
            query (str): The natural language query.

        Returns:
            Tuple[List[str], float]: List of identified tables and confidence score.
        """
        if not self.connection_manager or not self.connection_manager.is_connected():
            self.logger.error("Not connected to database")
            print("Not connected to database!")
            return [], 0.0
        
        if self.query_processor is None:
            self.logger.error("Query processor not initialized")
            print("Query processor not initialized. Please connect to the database.")
            return [], 0.0
        
        try:
            tables, confidence = self.query_processor.process_query(query)
            self.query_history.append(query)
            if len(self.query_history) > 10:
                self.query_history.pop(0)
            self.logger.debug(f"Query: {query}, Tables: {tables}, Confidence: {confidence}")
            return tables, confidence
        except Exception as e:
            self.logger.error(f"Query processing error: {e}")
            print(f"Query processing error: {e}")
            return [], 0.0

    def validate_tables_exist(self, tables: List[str]) -> Tuple[List[str], List[str]]:
        """Validate tables against the current schema.

        Args:
            tables (List[str]): List of table names (schema.table).

        Returns:
            Tuple[List[str], List[str]]: Valid and invalid table lists.
        """
        valid = []
        invalid = []
        schema_map = {s.lower(): s for s in self.schema_dict['tables']}
        
        for table in tables:
            parts = table.split('.')
            if len(parts) != 2:
                invalid.append(table)
                self.logger.warning(f"Invalid table format: {table}")
                continue
                
            schema, table_name = parts
            schema_lower = schema.lower()
            
            if (schema_lower in schema_map and 
                table_name.lower() in {t.lower() for t in self.schema_dict['tables'][schema_map[schema_lower]]}):
                valid.append(f"{schema_map[schema_lower]}.{table_name}")
            else:
                invalid.append(table)
                self.logger.warning(f"Table not found: {table}")
        
        self.logger.debug(f"Validated tables: Valid={valid}, Invalid={invalid}")
        return valid, invalid

    def generate_ddl(self, tables: List[str]):
        """Generate DDL statements for specified tables.

        Args:
            tables (List[str]): List of table names (schema.table).
        """
        for table in tables:
            if '.' not in table:
                print(f"Invalid format: {table}")
                self.logger.warning(f"Invalid table format for DDL: {table}")
                continue
                
            schema, table_name = table.split('.')
            if schema not in self.schema_dict['tables']:
                print(f"Schema not found: {schema}")
                self.logger.warning(f"Schema not found for DDL: {schema}")
                continue
                
            if table_name not in self.schema_dict['tables'][schema]:
                print(f"Table not found: {table_name} in schema {schema}")
                self.logger.warning(f"Table not found for DDL: {table_name} in {schema}")
                continue
                
            self.logger.debug(f"Generating DDL for {schema}.{table_name}")
            print(f"\n-- DDL for {schema}.{table_name}")
            columns = self.schema_dict['columns'][schema][table_name]
            col_defs = []
            
            for col_name, col_info in columns.items():
                col_def = f"    [{col_name}] {col_info['type']}"
                if col_info.get('is_primary_key'):
                    col_def += " PRIMARY KEY"
                if not col_info.get('nullable'):
                    col_def += " NOT NULL"
                col_defs.append(col_def)
                
            print("CREATE TABLE [{}].[{}] (\n{}\n);".format(
                schema, table_name, ",\n".join(col_defs)
            ))

    def close_connection(self):
        """Close the database connection."""
        if self.connection_manager:
            self.connection_manager.close()
            self.logger.info("Database connection closed")

    def is_connected(self) -> bool:
        """Check if the database connection is active.

        Returns:
            bool: True if connected, False otherwise.
        """
        return self.connection_manager and self.connection_manager.is_connected()

    def get_all_tables(self) -> List[str]:
        """Get a list of all tables in the schema.

        Returns:
            List[str]: List of table names (schema.table).
        """
        tables = []
        for schema in self.schema_dict['tables']:
            tables.extend(f"{schema}.{table}" for table in self.schema_dict['tables'][schema])
        self.logger.debug(f"All tables: {tables}")
        return tables

    def get_recent_queries(self, limit: int = 5) -> List[str]:
        """Get the most recent queries processed.

        Args:
            limit (int): Maximum number of queries to return.

        Returns:
            List[str]: List of recent queries.
        """
        return self.query_history[-limit:][::-1]

    def confirm_tables(self, query: str, tables: List[str]):
        """Confirm correct tables for a query and update feedback.

        Args:
            query (str): The natural language query.
            tables (List[str]): List of confirmed tables.
        """
        if self.feedback_manager:
            valid_tables, _ = self.validate_tables_exist(tables)
            if valid_tables:
                self.feedback_manager.store_feedback(query, valid_tables, self.schema_dict)
                if self.table_identifier:
                    self.table_identifier.update_weights_from_feedback(query, valid_tables)
                    if self.current_config:
                        self.table_identifier.save_model(f"models/{self.current_config['database']}_model.json")
                self.logger.info(f"Confirmed tables for query: {query}")
            else:
                self.logger.warning(f"No valid tables for feedback: {tables}")

    def update_feedback(self, query: str, tables: List[str]):
        """Update feedback with corrected tables.

        Args:
            query (str): The natural language query.
            tables (List[str]): List of corrected tables.
        """
        if self.feedback_manager:
            valid_tables, _ = self.validate_tables_exist(tables)
            if valid_tables:
                self.feedback_manager.store_feedback(query, valid_tables, self.schema_dict)
                if self.table_identifier:
                    self.table_identifier.update_weights_from_feedback(query, valid_tables)
                    if self.current_config:
                        self.table_identifier.save_model(f"models/{self.current_config['database']}_model.json")
                self.logger.info(f"Updated feedback for query: {query}")
            else:
                self.logger.warning(f"No valid tables for feedback update: {tables}")

    def clear_feedback(self):
        """Clear all feedback data."""
        if self.feedback_manager:
            self.feedback_manager.clear_feedback()
            print("Feedback cleared")
            self.logger.info("Feedback cleared")
        else:
            self.logger.error("Feedback manager not initialized")
            print("Feedback manager not initialized. Please connect to a database.")

if __name__ == "__main__":
    try:
        analyzer = DatabaseAnalyzer()
        analyzer.run()
    except Exception as e:
        logging.getLogger("main").error(f"Application failed: {e}")
        print(f"Application failed: {e}")