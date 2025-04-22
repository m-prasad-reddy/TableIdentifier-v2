import logging
import pandas as pd
import os
import numpy as np
import json
import csv
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class TableIdentifier:
    """Identifies relevant tables from natural language queries using NLP and feedback."""
    
    def __init__(self, schema_dict: Dict, feedback_manager, pattern_manager, name_match_manager, db_name: str):
        """Initialize with schema, feedback, pattern, and name match managers.

        Args:
            schema_dict: Schema dictionary from SchemaManager.
            feedback_manager: FeedbackManager instance.
            pattern_manager: PatternManager instance.
            name_match_manager: NameMatchManager instance.
            db_name: Name of the database.
        """
        self.logger = logging.getLogger("table_identifier")
        self.schema_dict = schema_dict
        self.feedback_manager = feedback_manager
        self.pattern_manager = pattern_manager
        self.name_match_manager = name_match_manager
        self.db_name = db_name
        self.training_data = []
        self.weights = {}
        self.embedder = None
        
        try:
            self.embedder = SentenceTransformer('all-distilroberta-v1')
            self.logger.debug("Loaded SentenceTransformer: all-distilroberta-v1")
        except Exception as e:
            self.logger.error(f"Error loading SentenceTransformer: {e}")
            self.embedder = None
        
        try:
            csv_path = os.path.join("app-config", "training_data.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path, quoting=csv.QUOTE_ALL, on_bad_lines='warn')
                self.training_data = df.values.tolist()
                self.logger.debug(f"Loaded {len(self.training_data)} training records from {csv_path}")
            else:
                self.logger.warning(f"Training CSV not found at {csv_path}")
        except Exception as e:
            self.logger.error(f"Error loading CSV: {str(e)}")
            self.training_data = []
        
        self._initialize_weights()
        self.logger.debug("Initialized TableIdentifier")

    def _initialize_weights(self):
        """Initialize weights for tables based on schema."""
        for schema in self.schema_dict["tables"]:
            for table in self.schema_dict["tables"][schema]:
                self.weights[f"{schema}.{table}"] = 1.0
        self.logger.debug(f"Initialized weights for {len(self.weights)} tables")

    def identify_tables(self, query: str) -> Tuple[List[str], float]:
        """Identify tables relevant to the query.

        Args:
            query: Natural language query.

        Returns:
            Tuple: List of table names (schema.table) and confidence score.
        """
        self.logger.debug(f"Identifying tables for query: {query}")
        try:
            # Check feedback for similar queries
            feedback = self.feedback_manager.get_similar_feedback(query, threshold=0.8)
            if feedback:
                self.logger.debug(f"Found similar feedback: {feedback['tables']}")
                return feedback['tables'], 0.9
            
            # Use name matching with synonyms
            matched_tables = self.name_match_manager.match_names(query, self.schema_dict)
            if matched_tables:
                self.logger.debug(f"Name matched tables: {matched_tables}")
                return matched_tables, 0.85
            
            # Use embeddings if available
            if self.embedder:
                query_embedding = self.embedder.encode([query])[0]
                table_scores = []
                for table in self.weights:
                    table_embedding = self.embedder.encode([table])[0]
                    score = cosine_similarity([query_embedding], [table_embedding])[0][0]
                    weighted_score = score * self.weights[table]
                    table_scores.append((table, weighted_score))
                
                table_scores.sort(key=lambda x: x[1], reverse=True)
                top_tables = [table for table, score in table_scores[:5]]
                confidence = max(score for table, score in table_scores[:5]) if table_scores else 0.0
                if top_tables and confidence >= 0.8:
                    self.logger.debug(f"Embedding-based tables: {top_tables}, confidence: {confidence}")
                    return top_tables, confidence
            
            # Fallback to pattern matching
            matched_tables = self.pattern_manager.match_pattern(query)
            if matched_tables:
                self.logger.debug(f"Pattern matched tables: {matched_tables}")
                return matched_tables, 0.8
            
            # Fallback to training data
            for training_query, *tables in self.training_data:
                if query.lower() in training_query.lower():
                    self.logger.debug(f"Training data matched tables: {tables}")
                    return tables, 0.7
            
            self.logger.warning(f"No tables identified for query: {query}")
            return [], 0.0
        except Exception as e:
            self.logger.error(f"Error identifying tables: {str(e)}")
            return [], 0.0

    def save_name_matches(self):
        """Save name matching data to disk."""
        try:
            matches_path = os.path.join("models", f"{self.db_name}_matches.json")
            with open(matches_path, 'w') as f:
                json.dump(self.weights, f, indent=2)
            self.logger.debug(f"Saved name matches to {matches_path}")
        except Exception as e:
            self.logger.error(f"Error saving name matches: {e}")

    def save_model(self, model_path: str):
        """Save the trained model.

        Args:
            model_path: Path to save the model.
        """
        try:
            model_data = {
                "weights": self.weights,
                "version": "1.0"
            }
            with open(model_path, 'w') as f:
                json.dump(model_data, f, indent=2)
            self.logger.debug(f"Saved model to {model_path}")
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")

    def update_weights_from_feedback(self, query: str, tables: List[str]):
        """Update identification weights based on feedback.

        Args:
            query: The query.
            tables: Confirmed tables.
        """
        try:
            for table in tables:
                if table in self.weights:
                    self.weights[table] *= 1.1
                    self.logger.debug(f"Increased weight for {table} to {self.weights[table]}")
                else:
                    self.weights[table] = 1.0
                    self.logger.debug(f"Initialized weight for new table {table}")
            
            for table in self.weights:
                if table not in tables:
                    self.weights[table] *= 0.95
                    self.logger.debug(f"Decreased weight for {table} to {self.weights[table]}")
        except Exception as e:
            self.logger.error(f"Error updating weights: {e}")