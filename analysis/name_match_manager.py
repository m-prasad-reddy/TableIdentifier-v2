import logging
import os
import json
from typing import Dict, List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class NameMatchManager:
    """Manages name matching and synonym persistence for table identification."""

    def __init__(self, db_name: str, embedder: SentenceTransformer):
        """Initialize with database name and shared SentenceTransformer.

        Args:
            db_name (str): Name of the database.
            embedder (SentenceTransformer): Shared SentenceTransformer instance.
        """
        self.logger = logging.getLogger("name_match_manager")
        self.db_name = db_name
        self.embedder = embedder
        self.synonyms = {}
        self.matches_path = os.path.join("models", f"{self.db_name}_synonyms.json")
        self._load_synonyms()
        self.logger.debug(f"Initialized NameMatchManager for {db_name}")

    def _load_synonyms(self):
        """Load synonym mappings from disk."""
        try:
            if os.path.exists(self.matches_path):
                with open(self.matches_path, 'r') as f:
                    self.synonyms = json.load(f)
                self.logger.debug(f"Loaded synonyms from {self.matches_path}")
            else:
                self.logger.debug(f"No synonym file found at {self.matches_path}")
        except Exception as e:
            self.logger.error(f"Error loading synonyms: {e}")
            self.synonyms = {}

    def match_names(self, query: str, schema_dict: Dict) -> List[str]:
        """Match query terms to table and column names using synonyms and embeddings.

        Args:
            query (str): The query text.
            schema_dict (Dict): Schema dictionary.

        Returns:
            List[str]: Matching table names (schema.table).
        """
        self.logger.debug(f"Matching names for query: {query}")
        if not self.embedder:
            self.logger.warning("No embedder available, skipping name matching")
            return []

        try:
            query_lower = query.lower()
            query_embedding = self.embedder.encode([query_lower], convert_to_tensor=True)[0]
            matches = set()

            # Check synonyms
            for term, tables in self.synonyms.items():
                if term in query_lower:
                    matches.update(tables)

            # Semantic matching with table and column names
            table_texts = []
            table_names = []
            for schema in schema_dict["tables"]:
                for table in schema_dict["tables"][schema]:
                    table_name = f"{schema}.{table}"
                    table_texts.append(f"{schema}.{table}")
                    table_names.append(table_name)
                    for col_name in schema_dict["columns"][schema][table]:
                        table_texts.append(col_name)
                        table_names.append(table_name)

            if table_texts:
                table_embeddings = self.embedder.encode(table_texts, convert_to_tensor=True)
                similarities = cosine_similarity([query_embedding.cpu().numpy()], table_embeddings.cpu().numpy())[0]
                for idx, score in enumerate(similarities):
                    if score > 0.7:  # Threshold for relevance
                        matches.add(table_names[idx])

            matches = list(matches)
            self.logger.debug(f"Name matches: {matches}")
            return matches
        except Exception as e:
            self.logger.error(f"Error in name matching: {e}")
            return []

    def save_synonyms(self):
        """Save synonym mappings to disk."""
        try:
            os.makedirs(os.path.dirname(self.matches_path), exist_ok=True)
            with open(self.matches_path, 'w') as f:
                json.dump(self.synonyms, f, indent=2)
            self.logger.debug(f"Saved synonyms to {self.matches_path}")
        except Exception as e:
            self.logger.error(f"Error saving synonyms: {e}")

    def update_synonyms(self, query: str, tables: List[str]):
        """Update synonym mappings based on feedback.

        Args:
            query (str): The query text.
            tables (List[str]): Confirmed tables.
        """
        try:
            query_lower = query.lower()
            if query_lower not in self.synonyms:
                self.synonyms[query_lower] = []
            self.synonyms[query_lower] = list(set(self.synonyms[query_lower] + tables))
            self.save_synonyms()
            self.logger.debug(f"Updated synonyms for query: {query_lower}, tables: {tables}")
        except Exception as e:
            self.logger.error(f"Error updating synonyms: {e}")