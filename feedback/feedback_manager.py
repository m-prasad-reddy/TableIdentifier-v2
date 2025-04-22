import logging
import os
import sqlite3
import json
import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class FeedbackManager:
    """Manages feedback storage and retrieval using SQLite for thread-safe operations."""
    
    def __init__(self, db_name: str):
        """Initialize with database name and logging.

        Args:
            db_name: Name of the database.
        """
        self.logger = logging.getLogger("feedback_manager")
        self.db_name = db_name
        self.feedback_dir = os.path.join("feedback_cache", db_name)
        os.makedirs(self.feedback_dir, exist_ok=True)
        self.db_path = os.path.join(self.feedback_dir, "feedback.db")
        self.embedder = None
        self.feedback_cache = []
        
        try:
            self.embedder = SentenceTransformer('all-distilroberta-v1')
            self.logger.debug("Loaded SentenceTransformer for feedback")
        except Exception as e:
            self.logger.error(f"Error loading SentenceTransformer: {e}")
            self.embedder = None
        
        self._init_db()
        self._load_feedback_cache()
        self.logger.debug(f"Initialized FeedbackManager for {db_name}")

    def _init_db(self):
        """Initialize SQLite database for feedback storage."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS feedback (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        query TEXT NOT NULL,
                        tables TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        embedding BLOB NOT NULL
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS query_counts (
                        query TEXT PRIMARY KEY,
                        count INTEGER NOT NULL
                    )
                """)
                conn.commit()
            self.logger.debug(f"Initialized SQLite database at {self.db_path}")
        except Exception as e:
            self.logger.error(f"Error initializing SQLite database: {e}")

    def _load_feedback_cache(self):
        """Load feedback data from SQLite database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, query, tables, timestamp, embedding FROM feedback")
                self.feedback_cache = [
                    {
                        "id": row[0],
                        "query": row[1],
                        "tables": json.loads(row[2]),
                        "timestamp": row[3],
                        "embedding": np.frombuffer(row[4], dtype=np.float32)
                    }
                    for row in cursor.fetchall()
                ]
            self.logger.debug(f"Loaded {len(self.feedback_cache)} feedback entries")
        except Exception as e:
            self.logger.error(f"Error loading feedback cache: {e}")
            self.feedback_cache = []

    def store_feedback(self, query: str, tables: List[str], schema_dict: Dict):
        """Store feedback for a query-table mapping.

        Args:
            query: The query string.
            tables: List of table names.
            schema_dict: Schema dictionary for validation.
        """
        try:
            if not tables or not query:
                self.logger.warning("Empty query or tables, skipping feedback storage")
                return
            
            # Validate tables
            valid_tables = []
            for table in tables:
                schema, table_name = table.split('.', 1)
                if schema in schema_dict["tables"] and table_name in schema_dict["tables"][schema]:
                    valid_tables.append(table)
                else:
                    self.logger.warning(f"Invalid table {table} in feedback")
            
            if not valid_tables:
                self.logger.warning("No valid tables in feedback")
                return
            
            # Generate embedding
            embedding = None
            if self.embedder:
                embedding = self.embedder.encode([query])[0]
            else:
                embedding = np.zeros(768, dtype=np.float32)
            
            with sqlite3.connect(self.db_path, timeout=5.0) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO feedback (query, tables, timestamp, embedding) VALUES (?, ?, datetime('now'), ?)",
                    (query, json.dumps(valid_tables), embedding.tobytes())
                )
                cursor.execute(
                    "INSERT OR REPLACE INTO query_counts (query, count) VALUES (?, COALESCE((SELECT count + 1 FROM query_counts WHERE query = ?), 1))",
                    (query, query)
                )
                conn.commit()
            
            self._load_feedback_cache()  # Refresh cache
            self.logger.debug(f"Stored feedback for query: {query}, tables: {valid_tables}")
        except Exception as e:
            self.logger.error(f"Error storing feedback: {e}")

    def get_similar_feedback(self, query: str, threshold: float = 0.7) -> Optional[Dict]:
        """Retrieve feedback for similar queries.

        Args:
            query: The query string.
            threshold: Similarity threshold for matching.

        Returns:
            Dict: Feedback data if similar query found, None otherwise.
        """
        try:
            if not self.feedback_cache or not self.embedder:
                self.logger.debug("No feedback cache or embedder available")
                return None
            
            query_embedding = self.embedder.encode([query])[0]
            similarities = [
                (entry, cosine_similarity([query_embedding], [entry["embedding"]])[0][0])
                for entry in self.feedback_cache
            ]
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            if similarities and similarities[0][1] >= threshold:
                entry = similarities[0][0]
                self.logger.debug(f"Found similar feedback for query: {query}, similarity: {similarities[0][1]}")
                return {
                    "query": entry["query"],
                    "tables": entry["tables"],
                    "timestamp": entry["timestamp"]
                }
            
            self.logger.debug(f"No similar feedback found for query: {query}")
            return None
        except Exception as e:
            self.logger.error(f"Error in get_similar_feedback: {e}")
            return None

    def get_top_queries(self, limit: int = 5) -> List[tuple]:
        """Get the most frequent queries.

        Args:
            limit: Number of queries to return.

        Returns:
            List of (query, count) tuples.
        """
        try:
            with sqlite3.connect(self.db_path, timeout=5.0) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT query, count FROM query_counts ORDER BY count DESC LIMIT ?",
                    (limit,)
                )
                top_queries = cursor.fetchall()
            self.logger.debug(f"Retrieved {len(top_queries)} top queries")
            return top_queries
        except Exception as e:
            self.logger.error(f"Error getting top queries: {e}")
            return []

    def clear_feedback(self):
        """Clear all feedback data."""
        try:
            with sqlite3.connect(self.db_path, timeout=5.0) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM feedback")
                cursor.execute("DELETE FROM query_counts")
                conn.commit()
            self.feedback_cache = []
            self.logger.info("Cleared all feedback")
        except Exception as e:
            self.logger.error(f"Error clearing feedback: {e}")

    def export_feedback(self, export_dir: str):
        """Export feedback data to a directory.

        Args:
            export_dir: Directory to export feedback files.
        """
        try:
            os.makedirs(export_dir, exist_ok=True)
            with sqlite3.connect(self.db_path, timeout=5.0) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, query, tables, timestamp FROM feedback")
                copied = False
                for row in cursor.fetchall():
                    id_, query, tables, timestamp = row
                    meta = {
                        "query": query,
                        "tables": json.loads(tables),
                        "timestamp": timestamp
                    }
                    meta_file = os.path.join(export_dir, f"feedback_{id_}_meta.json")
                    with open(meta_file, 'w') as f:
                        json.dump(meta, f, indent=2)
                    copied = True
                if copied:
                    self.logger.info(f"Exported feedback to {export_dir}")
                else:
                    self.logger.info("No feedback to export")
        except Exception as e:
            self.logger.error(f"Error exporting feedback: {e}")

    def import_feedback(self, import_dir: str):
        """Import feedback data from a directory.

        Args:
            import_dir: Directory containing feedback files.
        """
        try:
            if not os.path.exists(import_dir):
                self.logger.error(f"Import directory {import_dir} does not exist")
                return
            
            with sqlite3.connect(self.db_path, timeout=5.0) as conn:
                cursor = conn.cursor()
                copied = False
                for fname in os.listdir(import_dir):
                    if fname.endswith("_meta.json"):
                        with open(os.path.join(import_dir, fname)) as f:
                            meta = json.load(f)
                        if 'query' not in meta or 'tables' not in meta or 'timestamp' not in meta:
                            self.logger.warning(f"Skipping invalid feedback file: {fname}")
                            continue
                        
                        embedding = self.embedder.encode([meta['query']])[0] if self.embedder else np.zeros(768, dtype=np.float32)
                        cursor.execute(
                            "INSERT INTO feedback (query, tables, timestamp, embedding) VALUES (?, ?, ?, ?)",
                            (meta['query'], json.dumps(meta['tables']), meta['timestamp'], embedding.tobytes())
                        )
                        cursor.execute(
                            "INSERT OR REPLACE INTO query_counts (query, count) VALUES (?, COALESCE((SELECT count + 1 FROM query_counts WHERE query = ?), 1))",
                            (meta['query'], meta['query'])
                        )
                        copied = True
                conn.commit()
            
            if copied:
                self._load_feedback_cache()
                self.logger.info(f"Imported feedback from {import_dir}")
            else:
                self.logger.info("No valid feedback files to import")
        except Exception as e:
            self.logger.error(f"Error importing feedback: {e}")