import logging
import spacy
from typing import List, Tuple

class QueryProcessor:
    """Processes natural language queries for database table identification."""

    def __init__(self, table_identifier):
        """Initialize with table identifier.

        Args:
            table_identifier: TableIdentifier instance.
        """
        self.logger = logging.getLogger("query_processor")
        self.table_identifier = table_identifier
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            self.logger.error(f"Failed to load spacy model: {e}")
            self.nlp = None
        self.logger.debug("Initialized QueryProcessor")

    def preprocess_query(self, query: str) -> str:
        """Preprocess the query for analysis.

        Validates query relevance and language using spacy.

        Args:
            query: The query string.

        Returns:
            str: Preprocessed query or empty string if invalid.
        """
        self.logger.debug(f"Preprocessing query: {query}")
        try:
            if not query or query.isspace():
                self.logger.warning("Empty query")
                return ""

            # Basic cleaning
            query = query.strip().lower()

            # Semantic validation with spacy
            if not self.nlp:
                self.logger.warning("Spacy model not loaded, skipping semantic validation")
                return query

            doc = self.nlp(query)
            # Check for meaningful structure
            has_noun_chunk = any(chunk for chunk in doc.noun_chunks)
            has_verb = any(token.pos_ == "VERB" for token in doc)
            if not (has_noun_chunk or has_verb):
                self.logger.warning(f"Query lacks meaningful structure: {query}")
                return ""

            # Check if query is likely English
            # Count English-like tokens (basic heuristic)
            english_tokens = sum(1 for token in doc if token.is_alpha and token.lang_ == "en")
            if english_tokens < len([token for token in doc if token.is_alpha]) * 0.5:
                self.logger.warning(f"Query may not be in English: {query}")
                return ""

            # Remove stop words and non-alphabetic tokens
            query_tokens = [token.text for token in doc if not token.is_stop and token.is_alpha]
            preprocessed = " ".join(query_tokens)
            self.logger.debug(f"Preprocessed query: {preprocessed}")
            return preprocessed
        except Exception as e:
            self.logger.error(f"Error preprocessing query: {e}")
            return ""

    def process_query(self, query: str) -> Tuple[List[str], float]:
        """Process a natural language query to identify relevant tables.

        Args:
            query: The query string.

        Returns:
            Tuple: List of table names and confidence score.
        """
        self.logger.debug(f"Processing query: {query}")
        try:
            preprocessed = self.preprocess_query(query)
            if not preprocessed:
                self.logger.warning(f"Invalid query after preprocessing: {query}")
                return [], 0.0

            tables, confidence = self.table_identifier.identify_tables(preprocessed)
            self.logger.debug(f"Identified tables: {tables}, confidence: {confidence}")
            return tables, confidence
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            return [], 0.0