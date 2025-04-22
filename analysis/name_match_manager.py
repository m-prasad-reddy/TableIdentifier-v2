import os
import json
import numpy as np
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import logging.config
import filelock

class NameMatchManager:
    """Manages synonym matching for database column names.

    Uses embeddings to map query tokens to columns, supports dynamic and default
    synonyms, and integrates with CSV training data.
    """

    def __init__(self, db_name: str):
        """Initialize with database name.

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
        
        self.logger = logging.getLogger("name_match_manager")
        self.db_name = db_name
        self.model = SentenceTransformer('all-distilroberta-v1')
        self.dynamic_matches = {}
        self.default_matches = {}
        self.cache_dir = os.path.join("schema_cache", db_name)
        self.dynamic_path = os.path.join(self.cache_dir, "dynamic_matches.json")
        self.default_path = os.path.join(self.cache_dir, "default_matches.json")
        os.makedirs(self.cache_dir, exist_ok=True)
        self._load_matches()
        self.logger.debug(f"Initialized NameMatchManager for {db_name}")

    def _load_matches(self):
        """Load dynamic and default matches from cache."""
        # Load dynamic matches
        if os.path.exists(self.dynamic_path):
            try:
                with open(self.dynamic_path) as f:
                    self.dynamic_matches = json.load(f)
                self.logger.debug(f"Loaded dynamic matches from {self.dynamic_path}")
            except Exception as e:
                self.logger.error(f"Error loading dynamic matches: {e}")

        # Load default matches
        if os.path.exists(self.default_path):
            try:
                with open(self.default_path) as f:
                    self.default_matches = json.load(f)
                self.logger.debug(f"Loaded default matches from {self.default_path}")
            except Exception as e:
                self.logger.error(f"Error loading default matches: {e}")

    def save_dynamic(self):
        """Save dynamic matches to cache."""
        with filelock.FileLock(os.path.join(self.cache_dir, "dynamic_matches.lock")):
            try:
                with open(self.dynamic_path, 'w') as f:
                    json.dump(self.dynamic_matches, f, indent=2)
                self.logger.debug(f"Saved dynamic matches to {self.dynamic_path}")
            except Exception as e:
                self.logger.error(f"Error saving dynamic matches: {e}")

    def save_to_default(self):
        """Save dynamic matches to default matches."""
        with filelock.FileLock(os.path.join(self.cache_dir, "default_matches.lock")):
            try:
                self.default_matches.update(self.dynamic_matches)
                with open(self.default_path, 'w') as f:
                    json.dump(self.default_matches, f, indent=2)
                self.logger.debug(f"Saved default matches to {self.default_path}")
            except Exception as e:
                self.logger.error(f"Error saving default matches: {e}")

    def get_token_embeddings(self, tokens: List[str]) -> np.ndarray:
        """Generate embeddings for a list of tokens.

        Args:
            tokens (List[str]): List of tokens.

        Returns:
            np.ndarray: Array of token embeddings.
        """
        try:
            embeddings = self.model.encode(tokens)
            self.logger.debug(f"Generated embeddings for {len(tokens)} tokens")
            return embeddings
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            return np.array([])

    def update_synonyms(self, tokens: List[str], token_embeddings: np.ndarray, columns: List[str]):
        """Update synonym mappings for columns based on tokens.

        Args:
            tokens (List[str]): List of query tokens.
            token_embeddings (np.ndarray): Embeddings for the tokens.
            columns (List[str]): List of column names.
        """
        try:
            column_embeddings = self.model.encode(columns)
            similarities = cosine_similarity(token_embeddings, column_embeddings)
            
            for i, token in enumerate(tokens):
                max_sim = np.max(similarities[i])
                if max_sim > 0.7:
                    col_idx = np.argmax(similarities[i])
                    column = columns[col_idx]
                    self.dynamic_matches[token] = self.dynamic_matches.get(token, {})
                    self.dynamic_matches[token][column] = max_sim
                    self.logger.debug(f"Updated synonym: {token} -> {column} (sim={max_sim:.2f})")
            
            self.save_dynamic()
        except Exception as e:
            self.logger.error(f"Error updating synonyms: {e}")

    def get_column_score(self, column: str, token_embeddings: np.ndarray) -> float:
        """Calculate the similarity score for a column against token embeddings.

        Args:
            column (str): Column name.
            token_embeddings (np.ndarray): Embeddings for query tokens.

        Returns:
            float: Maximum similarity score.
        """
        try:
            col_embedding = self.model.encode([column]).reshape(1, -1)
            similarities = cosine_similarity(col_embedding, token_embeddings)[0]
            max_sim = max(similarities) if similarities.size > 0 else 0.0
            self.logger.debug(f"Column score for {column}: {max_sim:.2f}")
            return max_sim
        except Exception as e:
            self.logger.error(f"Error calculating column score: {e}")
            return 0.0

    def get_synonyms(self, token: str) -> Dict[str, float]:
        """Get synonym mappings for a token.

        Args:
            token (str): The query token.

        Returns:
            Dict[str, float]: Dictionary of column names and similarity scores.
        """
        synonyms = self.dynamic_matches.get(token, {})
        synonyms.update(self.default_matches.get(token, {}))
        self.logger.debug(f"Synonyms for {token}: {synonyms}")
        return synonyms