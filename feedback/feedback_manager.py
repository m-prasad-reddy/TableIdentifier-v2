import os
import json
import re
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import logging
import logging.config
import filelock

nlp = spacy.load("en_core_web_sm")

class FeedbackManager:
    """Manages feedback for query-table mappings.

    Stores and retrieves feedback to improve table identification accuracy
    using embeddings and pattern matching.
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
        
        self.logger = logging.getLogger("feedback")
        self.db_name = db_name
        self.model = SentenceTransformer('all-distilroberta-v1')
        self.feedback_dir = os.path.join("feedback_cache", db_name)
        os.makedirs(self.feedback_dir, exist_ok=True)
        self.feedback_cache = {}
        self.pattern_cache = {}
        self._load_feedback_cache()
        self.logger.debug(f"Initialized FeedbackManager for {db_name}")

    def _load_feedback_cache(self):
        """Load feedback from cache with file locking."""
        with filelock.FileLock(os.path.join(self.feedback_dir, "feedback.lock")):
            self.feedback_cache.clear()
            self.pattern_cache.clear()
            
            for fname in os.listdir(self.feedback_dir):
                if fname.endswith("_meta.json"):
                    try:
                        with open(os.path.join(self.feedback_dir, fname)) as f:
                            meta = json.load(f)
                            if 'query' not in meta or 'tables' not in meta or 'timestamp' not in meta:
                                self.logger.warning(f"Skipping invalid feedback file {fname}")
                                continue
                            normalized_tables = [t.lower() for t in meta.get('tables', [])]
                            query_lower = meta['query'].lower()
                            
                            self.feedback_cache[query_lower] = {
                                'query': meta['query'],
                                'tables': normalized_tables,
                                'timestamp': meta['timestamp'],
                                'count': meta.get('count', 1)
                            }
                            
                            pattern = self._extract_query_pattern(meta['query'])
                            if pattern not in self.pattern_cache:
                                self.pattern_cache[pattern] = {
                                    'tables': normalized_tables,
                                    'timestamp': meta['timestamp'],
                                    'count': meta.get('count', 1)
                                }
                            else:
                                self.pattern_cache[pattern]['count'] += meta.get('count', 1)
                        self.logger.debug(f"Loaded feedback file {fname}")
                    except Exception as e:
                        self.logger.error(f"Error loading feedback file {fname}: {e}")

    def _extract_query_pattern(self, query: str) -> str:
        """Extract a generalized pattern from a query.

        Args:
            query (str): The query string.

        Returns:
            str: The extracted pattern.
        """
        doc = nlp(query.lower())
        pattern = []
        skip_next = False
        
        for i, token in enumerate(doc):
            if skip_next:
                skip_next = False
                continue
                
            if token.text in ('=', '!=') and i > 0:
                pattern.append(f"{doc[i-1].lemma_}=[CONDITION]")
                skip_next = True
            elif token.like_num and re.match(r'\d{4}', token.text):
                pattern.append('[YEAR]')
            elif token.text.lower() in ('between', 'from', 'to') and i + 1 < len(doc) and doc[i+1].like_num:
                pattern.append('[DATE_RANGE]')
                skip_next = True
            elif token.like_num:
                pattern.append('[VALUE]')
            elif token.is_quote:
                pattern.append('[LITERAL]')
            else:
                pattern.append(token.lemma_)
        
        pattern_str = ' '.join(pattern)
        self.logger.debug(f"Extracted pattern: {pattern_str}")
        return pattern_str

    def store_feedback(self, query: str, correct_tables: List[str], schema_dict: Dict) -> bool:
        """Store feedback for a query with validated tables.

        Args:
            query (str): The query string.
            correct_tables (List[str]): List of correct tables.
            schema_dict (Dict): Schema dictionary for validation.

        Returns:
            bool: True if feedback is stored successfully, False otherwise.
        """
        with filelock.FileLock(os.path.join(self.feedback_dir, "feedback.lock")):
            valid_tables, invalid_tables = self.validate_tables(correct_tables, schema_dict)
            
            if invalid_tables:
                self.logger.warning(f"Invalid tables: {invalid_tables}")
                print(f"Invalid tables: {invalid_tables}")
                return False
                
            if not valid_tables:
                self.logger.warning(f"No valid tables for query: {query}")
                print(f"No valid tables provided for query: {query}")
                return False
                
            normalized_tables = [t.lower() for t in valid_tables]
            existing = self._find_exact_match(query)
            
            if existing:
                self._update_feedback(existing, normalized_tables, query)
            else:
                self._create_new_feedback(query, normalized_tables)
            
            self._load_feedback_cache()
            self.logger.info(f"Stored feedback for query: {query}, tables: {normalized_tables}")
            return True

    def validate_tables(self, tables: List[str], schema_dict: Dict) -> Tuple[List[str], List[str]]:
        """Validate tables against the schema.

        Args:
            tables (List[str]): List of table names (schema.table).
            schema_dict (Dict): Schema dictionary.

        Returns:
            Tuple[List[str], List[str]]: Valid and invalid table lists.
        """
        valid_tables = []
        invalid_tables = []
        
        schema_map = {s.lower(): s for s in schema_dict['tables']}
        table_maps = {
            s: {t.lower(): t for t in schema_dict['tables'][s]} 
            for s in schema_dict['tables']
        }
        
        for table in tables:
            parts = table.split('.')
            if len(parts) != 2:
                self.logger.debug(f"Invalid table format: {table}")
                invalid_tables.append(table)
                continue
                
            schema_part, table_part = parts
            schema_lower = schema_part.lower()
            table_lower = table_part.lower()
            
            if schema_lower not in schema_map:
                self.logger.debug(f"Schema not found: {schema_lower}")
                invalid_tables.append(table)
                continue
                
            actual_schema = schema_map[schema_lower]
            
            if table_lower not in table_maps[actual_schema]:
                self.logger.debug(f"Table not found: {table_lower} in schema: {actual_schema}")
                invalid_tables.append(table)
                continue
                
            actual_table = table_maps[actual_schema][table_lower]
            valid_tables.append(f"{actual_schema}.{actual_table}")
                
        self.logger.debug(f"Valid tables: {valid_tables}, Invalid: {invalid_tables}")
        return valid_tables, invalid_tables

    def _find_exact_match(self, query: str) -> Optional[str]:
        """Find existing feedback for a query.

        Args:
            query (str): The query string.

        Returns:
            Optional[str]: Feedback ID if found, None otherwise.
        """
        query_lower = query.lower()
        for fname in os.listdir(self.feedback_dir):
            if fname.endswith("_meta.json"):
                try:
                    with open(os.path.join(self.feedback_dir, fname)) as f:
                        meta = json.load(f)
                        if 'query' in meta and meta['query'].lower() == query_lower:
                            self.logger.debug(f"Found exact match for query: {query}")
                            return fname.replace("_meta.json", "")
                except Exception:
                    continue
        return None

    def _update_feedback(self, feedback_id: str, tables: List[str], query: str):
        """Update existing feedback entry.

        Args:
            feedback_id (str): ID of the feedback entry.
            tables (List[str]): Updated tables.
            query (str): The query string.
        """
        meta_path = os.path.join(self.feedback_dir, f"{feedback_id}_meta.json")
        try:
            with open(meta_path, 'r+') as f:
                meta = json.load(f)
                meta['tables'] = tables
                meta['timestamp'] = datetime.now().isoformat()
                meta['count'] = meta.get('count', 1) + 1
                f.seek(0)
                json.dump(meta, f)
                f.truncate()
            self.logger.debug(f"Updated feedback {feedback_id}")
        except Exception as e:
            self.logger.error(f"Error updating feedback {feedback_id}: {e}")

    def _create_new_feedback(self, query: str, tables: List[str]):
        """Create a new feedback entry.

        Args:
            query (str): The query string.
            tables (List[str]): Associated tables.
        """
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        try:
            embedding = self.model.encode(query)
            np.save(os.path.join(self.feedback_dir, f"{timestamp}_emb.npy"), embedding)
            with open(os.path.join(self.feedback_dir, f"{timestamp}_meta.json"), 'w') as f:
                json.dump({
                    'query': query,
                    'tables': tables,
                    'timestamp': datetime.now().isoformat(),
                    'count': 1
                }, f)
            self.logger.debug(f"Created new feedback for query: {query}")
        except Exception as e:
            self.logger.error(f"Error creating feedback for query {query}: {e}")

    def get_similar_feedback(self, query: str, threshold: float = 0.7) -> Optional[List[Dict]]:
        """Retrieve similar feedback based on query similarity.

        Args:
            query (str): The query string.
            threshold (float): Similarity threshold (default: 0.7).

        Returns:
            Optional[List[Dict]]: List of similar feedback entries, or None if none found.
        """
        with filelock.FileLock(os.path.join(self.feedback_dir, "feedback.lock")):
            try:
                query_lower = query.lower()
                if query_lower in self.feedback_cache and self.feedback_cache[query_lower]['tables']:
                    self.logger.debug(f"Exact feedback match for query: {query}")
                    return [{
                        'similarity': 1.0,
                        'query': self.feedback_cache[query_lower]['query'],
                        'tables': self.feedback_cache[query_lower]['tables'],
                        'timestamp': self.feedback_cache[query_lower]['timestamp'],
                        'type': 'exact',
                        'count': self.feedback_cache[query_lower]['count']
                    }]

                pattern = self._extract_query_pattern(query)
                if pattern in self.pattern_cache and self.pattern_cache[pattern]['tables']:
                    self.logger.debug(f"Pattern match for query: {query}")
                    return [{
                        'similarity': 1.0,
                        'query': query,
                        'tables': self.pattern_cache[pattern]['tables'],
                        'timestamp': self.pattern_cache[pattern]['timestamp'],
                        'type': 'pattern',
                        'pattern': pattern,
                        'count': self.pattern_cache[pattern]['count']
                    }]

                query_emb = self.model.encode(query).reshape(1, -1)
                feedback_items = []
                
                for fname in os.listdir(self.feedback_dir):
                    if fname.endswith("_emb.npy"):
                        emb_path = os.path.join(self.feedback_dir, fname)
                        meta_path = emb_path.replace("_emb.npy", "_meta.json")
                        
                        if os.path.exists(meta_path):
                            try:
                                with open(meta_path) as f:
                                    meta = json.load(f)
                                if not meta.get('tables') or 'query' not in meta:
                                    continue
                                stored_emb = np.load(emb_path)
                                similarity = cosine_similarity(query_emb, stored_emb.reshape(1, -1))[0][0]
                                
                                if similarity >= threshold:
                                    feedback_items.append({
                                        "similarity": similarity,
                                        "query": meta["query"],
                                        "tables": meta["tables"],
                                        "timestamp": meta["timestamp"],
                                        "type": "semantic",
                                        "count": meta.get('count', 1)
                                    })
                            except Exception:
                                continue
                
                feedback_items.sort(key=lambda x: x["similarity"], reverse=True)
                self.logger.debug(f"Similar feedback: {feedback_items}")
                return feedback_items if feedback_items else None
            
            except Exception as e:
                self.logger.error(f"Feedback retrieval error: {e}")
                return None

    def get_top_queries(self, n: int) -> List[Tuple[str, int]]:
        """Get the top N most frequent queries.

        Args:
            n (int): Number of queries to return.

        Returns:
            List[Tuple[str, int]]: List of (query, count) tuples.
        """
        top_queries = [
            (meta['query'], meta['count'])
            for query, meta in self.feedback_cache.items()
            if meta['tables'] and 'query' in meta
        ]
        top_queries.sort(key=lambda x: (-x[1], x[0]))
        self.logger.debug(f"Top {n} queries: {top_queries[:n]}")
        return top_queries[:n]

    def clear_feedback(self):
        """Clear all feedback data."""
        with filelock.FileLock(os.path.join(self.feedback_dir, "feedback.lock")):
            try:
                for fname in os.listdir(self.feedback_dir):
                    if fname.endswith(("_meta.json", "_emb.npy")):
                        os.remove(os.path.join(self.feedback_dir, fname))
                self._load_feedback_cache()
                self.logger.info("Feedback cleared")
            except Exception as e:
                self.logger.error(f"Error clearing feedback: {e}")