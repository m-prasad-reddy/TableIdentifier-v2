import logging
import os
import json
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class NameMatchManager:
    """Manages matching of query terms to schema entities with synonym persistence."""
    
    def __init__(self, db_name: str):
        """Initialize with database name and logging.

        Args:
            db_name: Name of the database.
        """
        self.logger = logging.getLogger("name_match_manager")
        self.db_name = db_name
        self.matches_path = os.path.join("app-config", f"{db_name}_dynamic_name_matches.json")
        self.synonyms = {}
        self.embedder = None
        
        try:
            self.embedder = SentenceTransformer('all-distilroberta-v1')
            self.logger.debug("Loaded SentenceTransformer for name matching")
        except Exception as e:
            self.logger.error(f"Error loading SentenceTransformer: {e}")
            self.embedder = None
        
        self._load_synonyms()
        self.logger.debug(f"Initialized NameMatchManager for {db_name}")

    def _load_synonyms(self):
        """Load synonym mappings from disk."""
        try:
            if os.path.exists(self.matches_path):
                with open(self.matches_path, 'r') as f:
                    self.synonyms = json.load(f)
                self.logger.debug(f"Loaded synonyms from {self.matches_path}")
        except Exception as e:
            self.logger.error(f"Error loading synonyms: {e}")
            self.synonyms = {}

    def _save_synonyms(self):
        """Save synonym mappings to disk."""
        try:
            with open(self.matches_path, 'w') as f:
                json.dump(self.synonyms, f, indent=2)
            self.logger.debug(f"Saved synonyms to {self.matches_path}")
        except Exception as e:
            self.logger.error(f"Error saving synonyms: {e}")

    def match_names(self, query: str, schema_dict: Dict) -> List[str]:
        """Match query terms to schema entities.

        Args:
            query: The query string.
            schema_dict: Schema dictionary.

        Returns:
            List of matched table names (schema.table).
        """
        self.logger.debug(f"Matching names for query: {query}")
        try:
            matched_tables = set()
            query_terms = query.lower().split()
            
            # Check synonyms first
            for term in query_terms:
                if term in self.synonyms:
                    matched_tables.update(self.synonyms[term])
                    self.logger.debug(f"Synonym matched: {term} -> {self.synonyms[term]}")
            
            # Use embeddings for unmatched terms
            if self.embedder and not matched_tables:
                query_embedding = self.embedder.encode([query])[0]
                for schema in schema_dict["tables"]:
                    for table in schema_dict["tables"][schema]:
                        table_name = f"{schema}.{table}"
                        table_embedding = self.embedder.encode([table_name])[0]
                        score = cosine_similarity([query_embedding], [table_embedding])[0][0]
                        if score >= 0.8:
                            matched_tables.add(table_name)
                            self.synonyms.setdefault(term, []).append(table_name)
                            self.logger.debug(f"Embedding matched: {term} -> {table_name}, score: {score}")
                
                self._save_synonyms()
            
            return list(matched_tables)
        except Exception as e:
            self.logger.error(f"Error matching names: {str(e)}")
            return []