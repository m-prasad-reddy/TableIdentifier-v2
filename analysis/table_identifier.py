import logging
import pandas as pd
import os
import numpy as np
import json
import csv
import spacy
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class TableIdentifier:
    """Identifies relevant tables from natural language queries using NLP and feedback."""

    def __init__(self, schema_dict: Dict, feedback_manager, pattern_manager, name_match_manager, db_name: str, embedder: SentenceTransformer):
        """Initialize with schema, feedback, pattern, name match managers, and shared embedder.

        Args:
            schema_dict: Schema dictionary from SchemaManager.
            feedback_manager: FeedbackManager instance.
            pattern_manager: PatternManager instance.
            name_match_manager: NameMatchManager instance.
            db_name: Name of the database.
            embedder: Shared SentenceTransformer instance.
        """
        self.logger = logging.getLogger("table_identifier")
        self.schema_dict = schema_dict
        self.feedback_manager = feedback_manager
        self.pattern_manager = pattern_manager
        self.name_match_manager = name_match_manager
        self.db_name = db_name
        self.embedder = embedder
        self.weights = {}
        self.table_embeddings = {}
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            self.logger.error(f"Failed to load spacy model: {e}")
            self.nlp = None

        # Load training data
        try:
            csv_path = os.path.join("app-config", "training_data.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path, quoting=csv.QUOTE_ALL, on_bad_lines='warn')
                self.training_data = df.values.tolist()
                self.logger.debug(f"Loaded {len(self.training_data)} training records from {csv_path}")
            else:
                self.logger.warning(f"Training CSV not found at {csv_path}")
                self.training_data = []
        except Exception as e:
            self.logger.error(f"Error loading CSV: {e}")
            self.training_data = []

        self._initialize_weights()
        self._cache_table_embeddings()
        self.logger.debug("Initialized TableIdentifier")

    def _initialize_weights(self):
        """Initialize weights for tables based on schema."""
        for schema in self.schema_dict["tables"]:
            for table in self.schema_dict["tables"][schema]:
                self.weights[f"{schema}.{table}"] = 1.0
        self.logger.debug(f"Initialized weights for {len(self.weights)} tables")

    def _cache_table_embeddings(self):
        """Cache embeddings for table and column metadata to optimize performance."""
        if not self.embedder:
            self.logger.warning("No embedder available, skipping table embedding caching")
            return

        try:
            table_texts = []
            table_names = []
            for schema in self.schema_dict["tables"]:
                for table in self.schema_dict["tables"][schema]:
                    table_name = f"{schema}.{table}"
                    table_texts.append(f"{schema}.{table}")
                    table_names.append(table_name)
                    for col_name in self.schema_dict["columns"][schema][table]:
                        table_texts.append(f"{schema}.{table}.{col_name}")
                        table_names.append(table_name)

            if table_texts:
                embeddings = self.embedder.encode(table_texts, convert_to_tensor=True)
                for table_name, embedding in zip(table_names, embeddings):
                    if table_name not in self.table_embeddings:
                        self.table_embeddings[table_name] = [embedding]
                    else:
                        self.table_embeddings[table_name].append(embedding)
                self.logger.debug(f"Cached embeddings for {len(table_names)} table/column metadata entries")
        except Exception as e:
            self.logger.error(f"Error caching table embeddings: {e}")
            self.table_embeddings = {}

    def identify_tables(self, query: str) -> Tuple[List[str], float]:
        """Identify tables relevant to the query.

        Combines feedback, pattern matching, semantic embeddings, and keyword matching.

        Args:
            query: Natural language query.

        Returns:
            Tuple: List of table names (schema.table) and confidence score.
        """
        self.logger.debug(f"Identifying tables for query: {query}")
        try:
            # Step 1: Check feedback for similar queries
            feedback = self.feedback_manager.get_similar_feedback(query, threshold=0.8)
            if feedback:
                self.logger.debug(f"Found similar feedback: {feedback['tables']}")
                return feedback['tables'], 0.9

            # Step 2: Pattern matching
            pattern_matches = self.pattern_manager.match_pattern(query)
            if pattern_matches:
                self.logger.debug(f"Pattern matched tables: {pattern_matches}")
                return pattern_matches, 0.8

            # Step 3: Semantic matching with embeddings
            if self.embedder and self.table_embeddings:
                query_embedding = self.embedder.encode([query], convert_to_tensor=True)[0]
                table_scores = {}
                for table, embeddings in self.table_embeddings.items():
                    for emb in embeddings:
                        score = cosine_similarity([query_embedding.cpu().numpy()], [emb.cpu().numpy()])[0][0]
                        weighted_score = score * self.weights.get(table, 1.0)
                        if table not in table_scores or weighted_score > table_scores[table]:
                            table_scores[table] = weighted_score

                sorted_scores = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)
                top_tables = [table for table, score in sorted_scores[:3] if score > 0.5]
                confidence = max(score for table, score in sorted_scores[:3]) if sorted_scores else 0.0
                if top_tables:
                    self.logger.debug(f"Embedding-based tables: {top_tables}, confidence: {confidence}")
                    return top_tables, confidence

            # Step 4: Keyword matching (fallback)
            if self.nlp:
                doc = self.nlp(query.lower())
                keyword_matches = set()
                for schema in self.schema_dict["tables"]:
                    for table in self.schema_dict["tables"][schema]:
                        table_name = table.lower()
                        full_table = f"{schema}.{table}"
                        if table_name in query.lower() or table_name.replace('_', ' ') in query.lower():
                            keyword_matches.add(full_table)
                        for col_name in self.schema_dict["columns"][schema][table]:
                            col_lower = col_name.lower()
                            if col_lower in query.lower() or col_lower.replace('_', ' ') in query.lower():
                                keyword_matches.add(full_table)
                if keyword_matches:
                    self.logger.debug(f"Keyword matched tables: {keyword_matches}")
                    return list(keyword_matches), 0.7

            # Step 5: Fallback to training data
            for training_query, *tables in self.training_data:
                if query.lower() in training_query.lower():
                    self.logger.debug(f"Training data matched tables: {tables}")
                    return tables, 0.6

            self.logger.warning(f"No tables identified for query: {query}")
            return [], 0.0
        except Exception as e:
            self.logger.error(f"Error identifying tables: {e}")
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