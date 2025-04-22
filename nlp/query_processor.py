import logging
import spacy
from langdetect import detect, DetectorFactory
from typing import List, Dict, Tuple

# Ensure consistent language detection
DetectorFactory.seed = 0

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

        Args:
            query: The query string.

        Returns:
            str: Preprocessed query or empty string if invalid.
        """
        try:
            if not query or query.isspace():
                self.logger.warning("Empty query")
                return ""
                
            # Check if query is in English
            try:
                lang = detect(query)
                if lang != 'en':
                    self.logger.warning(f"Non-English query: {query} (language: {lang})")
                    return ""
            except Exception as e:
                self.logger.error(f"Error detecting language: {e}")
                return ""
                
            # Basic cleaning
            query = query.strip().lower()
            
            # Semantic validation
            if self.nlp:
                doc = self.nlp(query)
                if not any(chunk for chunk in doc.noun_chunks) and not any(token.pos_ == "VERB" for token in doc):
                    self.logger.warning(f"Query lacks meaningful structure: {query}")
                    return ""
            
            return query
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