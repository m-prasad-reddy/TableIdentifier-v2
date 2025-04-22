import os
import spacy
from spacy.matcher import Matcher
from typing import Dict
import logging
import logging.config

class NLPPipeline:
    """Processes natural language queries for SQL generation using spaCy.

    This class handles tokenization, entity recognition, and pattern matching
    to analyze queries and extract relevant information.
    """

    def __init__(self, pattern_manager, db_name: str = "BikeStores"):
        """Initialize the NLP pipeline with a pattern manager.

        Args:
            pattern_manager: The PatternManager instance for query patterns.
            db_name (str): Name of the database (default: "BikeStores").
        """
        # Ensure logs directory exists
        os.makedirs("logs", exist_ok=True)
        logging_config_path = "app-config/logging_config.ini"
        if os.path.exists(logging_config_path):
            try:
                logging.config.fileConfig(logging_config_path, disable_existing_loggers=False)
            except Exception as e:
                print(f"Error loading logging config: {e}")
        
        self.logger = logging.getLogger("nlp_pipeline")
        self.nlp = spacy.load("en_core_web_trf")
        self.matcher = Matcher(self.nlp.vocab)
        self.pattern_manager = pattern_manager
        self._load_patterns()
        self.logger.debug("Initialized NLPPipeline")

    def _load_patterns(self):
        """Load spaCy patterns from the PatternManager."""
        self.logger.debug("Loading patterns")
        patterns = self.pattern_manager.get_patterns()
        for query_string, table_weights in patterns.items():
            # Convert query string to spaCy pattern
            tokens = query_string.lower().split()
            spacy_pattern = [{"LOWER": token} for token in tokens]
            for table in table_weights:
                try:
                    self.matcher.add(f"TABLE_{table}", [spacy_pattern])
                    self.logger.debug(f"Added pattern '{query_string}' for table '{table}'")
                except Exception as e:
                    self.logger.error(f"Error adding pattern '{query_string}' for '{table}': {e}")
                    raise
        self.logger.debug("Patterns loaded")

    def analyze_query(self, query: str) -> Dict:
        """Analyze a query using spaCy to extract linguistic features.

        Args:
            query (str): The natural language query.

        Returns:
            Dict: Analysis results including entities, tokens, matches, dependencies, and relations.
        """
        self.logger.debug(f"Analyzing query: {query}")
        doc = self.nlp(query.lower())
        matches = self.matcher(doc)
        
        result = {
            "entities": [(ent.text, ent.label_) for ent in doc.ents],
            "tokens": [token.lemma_ for token in doc if not token.is_stop],
            "matches": [(self.nlp.vocab.strings[m_id], doc[start:end].text) 
                        for m_id, start, end in matches],
            "dependencies": [(token.text, token.dep_, token.head.text) for token in doc],
            "relations": [(token.text, token.dep_, token.head.text) for token in doc if token.dep_ in ('nsubj', 'dobj', 'pobj')]
        }
        self.logger.debug(f"Analysis result: {result}")
        return result