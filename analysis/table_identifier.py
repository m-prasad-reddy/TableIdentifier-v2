import spacy
from typing import Dict, List, Optional, Tuple
import json
import os
import logging
import logging.config
from sentence_transformers import SentenceTransformer
from analysis.name_match_manager import NameMatchManager
import pandas as pd
import filelock
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

nlp = spacy.load("en_core_web_sm")

class TableIdentifier:
    """Identifies database tables in natural language queries.

    Uses NLP, embeddings, and feedback to map queries to relevant tables,
    with support for CSV-based training.
    """

    def __init__(self, schema_dict: Dict, feedback_manager, pattern_manager):
        """Initialize with schema, feedback, and pattern managers.

        Args:
            schema_dict (Dict): Schema dictionary with table and column information.
            feedback_manager: FeedbackManager instance for query feedback.
            pattern_manager: PatternManager instance for query patterns.
        """
        # Ensure logs directory exists
        os.makedirs("logs", exist_ok=True)
        logging_config_path = "app-config/logging_config.ini"
        if os.path.exists(logging_config_path):
            try:
                logging.config.fileConfig(logging_config_path, disable_existing_loggers=False)
            except Exception as e:
                print(f"Error loading logging config: {e}")
        
        self.logger = logging.getLogger("table_identifier")
        self.schema_dict = schema_dict
        self.feedback_manager = feedback_manager
        self.pattern_manager = pattern_manager
        self.model = SentenceTransformer('all-distilroberta-v1')
        self.name_match_manager = NameMatchManager(feedback_manager.db_name)
        self.weights = self._load_weights()
        self.csv_path = os.path.join("app-config", feedback_manager.db_name, "db_config_trainer.csv")
        self._initialize_from_csv()
        self.logger.debug("Initialized TableIdentifier")

    def _load_weights(self) -> Dict:
        """Load table weights from cache.

        Returns:
            Dict: Dictionary of table weights.
        """
        weights_path = os.path.join("schema_cache", self.feedback_manager.db_name, "weights.json")
        if os.path.exists(weights_path):
            try:
                with open(weights_path) as f:
                    self.logger.debug(f"Loaded weights from {weights_path}")
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading weights: {e}")
        return {}

    def _save_weights(self):
        """Save table weights to cache."""
        weights_path = os.path.join("schema_cache", self.feedback_manager.db_name, "weights.json")
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        try:
            with open(weights_path, 'w') as f:
                json.dump(self.weights, f)
            self.logger.debug(f"Saved weights to {weights_path}")
        except Exception as e:
            self.logger.error(f"Error saving weights: {e}")

    def _initialize_from_csv(self):
        """Initialize patterns and weights from a CSV training file."""
        if not os.path.exists(self.csv_path):
            self._create_template_csv()
        
        try:
            df = pd.read_csv(self.csv_path)
            for _, row in df.iterrows():
                schema = row['Schema']
                table = row['Table_Name']
                table_full = f"{schema}.{table}"
                description = str(row['Description']).lower().split()
                columns = str(row['Columns_List']).split(',')
                
                # Update patterns
                for token in description:
                    self.pattern_manager.pattern_weights[token] = self.pattern_manager.pattern_weights.get(token, {})
                    self.pattern_manager.pattern_weights[token][table_full] = 0.5
                
                # Update name matches
                token_embeddings = self.name_match_manager.get_token_embeddings(description)
                self.name_match_manager.update_synonyms(description, token_embeddings, columns)
                
                # Update weights
                self.weights[table_full] = self.weights.get(table_full, {})
                for token in description:
                    self.weights[table_full][token] = self.weights[table_full].get(token, 0.0) + 0.1
            
            self._save_weights()
            self.name_match_manager.save_dynamic()
            self.pattern_manager.save_patterns()
            self.logger.debug(f"Initialized from CSV: {self.csv_path}")
        except Exception as e:
            self.logger.error(f"Error loading CSV: {e}")

    def _create_template_csv(self):
        """Create a template CSV file for training data."""
        columns = [
            'DB_Config', 'Schema', 'Table_Name', 'Primary_Keys', 'Foreign_Keys',
            'Associated_Tables', 'Associated_Views', 'Description', 'Columns_List'
        ]
        df = pd.DataFrame(columns=columns)
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        df.to_csv(self.csv_path, index=False)
        self.logger.debug(f"Created template CSV: {self.csv_path}")

    def save_name_matches(self):
        """Save dynamic name matches to default storage."""
        self.name_match_manager.save_to_default()
        self.logger.debug("Saved name matches")

    def identify_tables(self, query: str) -> Tuple[Optional[List[str]], bool]:
        """Identify tables in a query using feedback and NLP.

        Args:
            query (str): The natural language query.

        Returns:
            Tuple[Optional[List[str]], bool]: List of identified tables and confidence flag.
        """
        self.logger.debug(f"Identifying tables for query: {query}")
        with filelock.FileLock(os.path.join(self.feedback_manager.feedback_dir, "feedback.lock")):
            feedback = self.feedback_manager.get_similar_feedback(query, threshold=0.7)
            if feedback and feedback[0]['tables']:
                tables = feedback[0]['tables']
                valid_tables, _ = self.feedback_manager.validate_tables(tables, self.schema_dict)
                if valid_tables:
                    self.logger.info(f"Used feedback tables: {valid_tables}")
                    return valid_tables, True
        return self._identify_tables_nlp(query)

    def _identify_tables_nlp(self, query: str) -> Tuple[Optional[List[str]], bool]:
        """Identify tables using NLP techniques.

        Args:
            query (str): The natural language query.

        Returns:
            Tuple[Optional[List[str]], bool]: List of identified tables and confidence flag.
        """
        try:
            doc = nlp(query.lower())
            table_scores = {}
            query_embedding = self.model.encode(query).reshape(1, -1)
            token_embeddings = self.name_match_manager.get_token_embeddings([t.lemma_ for t in doc])
            
            for schema in self.schema_dict['tables']:
                for table in self.schema_dict['tables'][schema]:
                    table_full = f"{schema}.{table}"
                    score = 0.0
                    
                    # Direct table name match
                    if table.lower() in query.lower():
                        score += 0.5
                    
                    # Column similarity
                    for col in self.schema_dict['columns'][schema][table]:
                        col_score = self.name_match_manager.get_column_score(col, token_embeddings)
                        score += col_score * 0.8
                    
                    # Pattern weight
                    score += self.pattern_manager.get_pattern_weight(query, table_full)
                    
                    # Token weights
                    for token in doc:
                        lemma = token.lemma_.lower()
                        score += self.weights.get(table_full, {}).get(lemma, 0.0)
                    
                    # Semantic similarity with table description
                    if table_full in self.weights:
                        table_desc = ' '.join(self.weights[table_full].keys())
                        if table_desc:
                            table_embedding = self.model.encode(table_desc).reshape(1, -1)
                            similarity = cosine_similarity(query_embedding, table_embedding)[0][0]
                            score += similarity * 0.3
                    
                    if score > 0:
                        table_scores[table_full] = score
            
            sorted_tables = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            selected_tables = [table for table, _ in sorted_tables]
            
            confidence = bool(selected_tables and max(table_scores.values(), default=0) > 0.5)
            self.logger.debug(f"Tables: {selected_tables}, Confidence: {confidence}")
            return selected_tables or None, confidence
        
        except Exception as e:
            self.logger.error(f"NLP error: {e}")
            return None, False

    def update_weights_from_feedback(self, query: str, tables: List[str]):
        """Update weights based on user feedback.

        Args:
            query (str): The query string.
            tables (List[str]): List of confirmed tables.
        """
        self.logger.debug(f"Updating weights for query: {query}, Tables: {tables}")