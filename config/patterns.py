import json
import os
import re
import spacy
from typing import Dict, List
import logging
import logging.config

class PatternManager:
    """Manages query patterns for table identification.

    Loads and stores patterns from a JSON file and provides pattern-based matching
    against schema metadata using NLP techniques.
    """

    def __init__(self, schema_dict: Dict):
        """Initialize with a schema dictionary.

        Args:
            schema_dict (Dict): Schema dictionary containing table and column information.
        """
        # Ensure logs directory exists
        os.makedirs("logs", exist_ok=True)
        logging_config_path = "app-config/logging_config.ini"
        if os.path.exists(logging_config_path):
            try:
                logging.config.fileConfig(logging_config_path, disable_existing_loggers=False)
            except Exception as e:
                self.logger = logging.getLogger("patterns")
                self.logger.error(f"Error loading logging config: {e}")
        
        self.logger = logging.getLogger("patterns")
        self.schema_dict = schema_dict
        self.pattern_weights = self._load_patterns()
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            self.logger.error(f"Failed to load spacy model: {e}")
            self.nlp = None
        self.logger.debug(f"Initialized PatternManager with {len(self.pattern_weights)} patterns")

    def _load_patterns(self) -> Dict[str, Dict[str, float]]:
        """Load patterns from global_patterns.json.

        Returns:
            Dict[str, Dict[str, float]]: Dictionary of query patterns and table weights.
        """
        pattern_path = "app-config/global_patterns.json"
        patterns = {}
        try:
            if os.path.exists(pattern_path):
                with open(pattern_path, 'r') as f:
                    patterns = json.load(f)
                self.logger.debug(f"Loaded patterns from {pattern_path}")
            else:
                self.logger.warning(f"Pattern file not found at {pattern_path}")
        except Exception as e:
            self.logger.error(f"Error loading patterns: {e}")
        
        # Normalize patterns
        normalized = {}
        for query, weights in patterns.items():
            norm_query = re.sub(r'\s+', ' ', query.lower().strip())
            normalized[norm_query] = {
                table: float(weight) for table, weight in weights.items()
            }
        return normalized

    def match_pattern(self, query: str) -> List[str]:
        """Match query against patterns and schema metadata.

        Uses spacy for tokenization and entity recognition to match query terms
        against table names, column names, and value-based patterns (e.g., dates, locations).

        Args:
            query (str): The query text.

        Returns:
            List[str]: Matching table names (schema.table).
        """
        self.logger.debug(f"Matching patterns for query: {query}")
        if not self.nlp:
            self.logger.warning("Spacy model not loaded, skipping pattern matching")
            return []

        try:
            query_lower = query.lower()
            doc = self.nlp(query_lower)
            matches = set()

            # Keyword matching against table and column names
            for schema in self.schema_dict['tables']:
                for table in self.schema_dict['tables'][schema]:
                    table_name = table.lower()
                    full_table = f"{schema}.{table}"
                    # Match table name
                    if table_name in query_lower or table_name.replace('_', ' ') in query_lower:
                        matches.add(full_table)
                    # Match column names
                    for col_name in self.schema_dict['columns'][schema][table]:
                        col_lower = col_name.lower()
                        if col_lower in query_lower or col_lower.replace('_', ' ') in query_lower:
                            matches.add(full_table)

            # Pattern matching for entities (dates, locations, numbers)
            for token in doc:
                if token.ent_type_ == "DATE":
                    # Match tables with date columns
                    for schema in self.schema_dict['tables']:
                        for table in self.schema_dict['tables'][schema]:
                            for col_name, col_info in self.schema_dict['columns'][schema][table].items():
                                if col_info['type'].lower() in ['date', 'datetime', 'timestamp']:
                                    matches.add(f"{schema}.{table}")
                                    break
                if token.ent_type_ == "GPE":  # Geographic entities (e.g., India, USA)
                    # Match tables likely to contain location data
                    for schema in self.schema_dict['tables']:
                        for table in self.schema_dict['tables'][schema]:
                            for col_name in self.schema_dict['columns'][schema][table]:
                                if 'city' in col_name.lower() or 'country' in col_name.lower() or 'state' in col_name.lower():
                                    matches.add(f"{schema}.{table}")
                                    break
                if token.ent_type_ == "CARDINAL":  # Numbers
                    # Match tables with numeric columns
                    for schema in self.schema_dict['tables']:
                        for table in self.schema_dict['tables'][schema]:
                            for col_name, col_info in self.schema_dict['columns'][schema][table].items():
                                if col_info['type'].lower() in ['int', 'integer', 'numeric', 'decimal', 'float']:
                                    matches.add(f"{schema}.{table}")
                                    break

            # Check pattern weights from global_patterns.json
            norm_query = re.sub(r'\s+', ' ', query_lower.strip())
            if norm_query in self.pattern_weights:
                for table, weight in self.pattern_weights[norm_query].items():
                    if weight > 0.5:  # Threshold for relevance
                        matches.add(table)

            matches = list(matches)
            self.logger.debug(f"Pattern matches: {matches}")
            return matches
        except Exception as e:
            self.logger.error(f"Error in pattern matching: {e}")
            return []

    def get_patterns(self) -> Dict[str, Dict[str, float]]:
        """Return the loaded patterns.

        Returns:
            Dict[str, Dict[str, float]]: Dictionary of query patterns and table weights.
        """
        return self.pattern_weights

    def get_pattern_weight(self, query: str, table: str) -> float:
        """Get the weight for a query-table pair.

        Args:
            query (str): The query string.
            table (str): The table name (schema.table).

        Returns:
            float: The weight for the query-table pair, or 0.0 if not found.
        """
        norm_query = re.sub(r'\s+', ' ', query.lower().strip())
        return self.pattern_weights.get(norm_query, {}).get(table, 0.0)

    def save_patterns(self):
        """Save patterns to global_patterns.json."""
        pattern_path = "app-config/global_patterns.json"
        try:
            with open(pattern_path, 'w') as f:
                json.dump(self.pattern_weights, f, indent=2)
            self.logger.debug(f"Saved patterns to {pattern_path}")
        except Exception as e:
            self.logger.error(f"Error saving patterns: {e}")