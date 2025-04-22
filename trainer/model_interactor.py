import json
import spacy
import logging
import logging.config
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

nlp = spacy.load("en_core_web_sm")

class TableIdentificationModel:
    """Standalone model for table identification from natural language queries.

    Loads a trained model and identifies tables without requiring a live database connection.
    """

    def __init__(self, model_path: str):
        """Initialize with model path.

        Args:
            model_path (str): Path to the trained model file.
        """
        # Ensure logs directory exists
        os.makedirs("logs", exist_ok=True)
        logging_config_path = "app-config/logging_config.ini"
        if os.path.exists(logging_config_path):
            try:
                logging.config.fileConfig(logging_config_path, disable_existing_loggers=False)
            except Exception as e:
                print(f"Error loading logging config: {e}")
        
        self.logger = logging.getLogger("trainer")
        self.model = SentenceTransformer('all-distilroberta-v1')
        self.weights = {}
        self.schema_dict = {}
        self.dynamic_matches = {}
        self.default_matches = {}
        self.load_model(model_path)
        self.logger.debug(f"Initialized TableIdentificationModel with {model_path}")

    def load_model(self, model_path: str):
        """Load the trained model from a JSON file.

        Args:
            model_path (str): Path to the model file.

        Raises:
            Exception: If loading fails.
        """
        try:
            with open(model_path, 'r') as f:
                model_data = json.load(f)
            self.weights = model_data['weights']
            self.schema_dict = model_data['schema_dict']
            self.dynamic_matches = model_data['dynamic_matches']
            self.default_matches = model_data['default_matches']
            self.logger.debug(f"Loaded model from {model_path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise

    def identify_tables(self, query: str) -> list[str] | None:
        """Identify tables for a query using the loaded model.

        Args:
            query (str): The natural language query.

        Returns:
            list[str] | None: List of identified tables, or None if none are found.
        """
        self.logger.debug(f"Identifying tables for query: {query}")
        try:
            doc = nlp(query.lower())
            table_scores = {}
            query_embedding = self.model.encode(query).reshape(1, -1)
            tokens = [t.lemma_ for t in doc]
            token_embeddings = self.model.encode(tokens)
            
            for schema in self.schema_dict['tables']:
                for table in self.schema_dict['tables'][schema]:
                    table_full = f"{schema}.{table}"
                    score = 0.0
                    
                    # Direct table name match
                    if table.lower() in query.lower():
                        score += 0.5
                    
                    # Column similarity
                    for col in self.schema_dict['columns'][schema][table]:
                        col_embedding = self.model.encode([col]).reshape(1, -1)
                        similarities = cosine_similarity(col_embedding, token_embeddings)[0]
                        score += max(similarities) * 0.8 if max(similarities) > 0.7 else 0.0
                    
                    # Token weights
                    for token in doc:
                        lemma = token.lemma_.lower()
                        score += self.weights.get(table_full, {}).get(lemma, 0.0)
                    
                    # Semantic similarity with table description
                    if table_full in self.weights:
                        table_desc = ' '.join(self.weights[table_full].keys())
                        if table_desc:
                            table_embedding = self.model.encode(table_desc).reshape(1, -1)
                            similarity = cosine_similarity(query_embedding, table_embedding)[0][0]
                            score += similarity * 0.3
                    
                    if score > 0:
                        table_scores[table_full] = score
            
            sorted_tables = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            selected_tables = [table for table, _ in sorted_tables]
            
            if selected_tables and max(table_scores.values(), default=0) > 0.5:
                self.logger.debug(f"Identified tables: {selected_tables}")
                return selected_tables
            else:
                self.logger.debug("No relevant tables identified")
                return None
        
        except Exception as e:
            self.logger.error(f"Error identifying tables: {e}")
            return None

def main():
    """Run the model interactor CLI."""
    model_path = input("Enter model path [default: models/table_identifier_model.json]: ").strip()
    if not model_path:
        model_path = "models/table_identifier_model.json"
    
    try:
        model = TableIdentificationModel(model_path)
        print("\n=== Table Identification Model ===")
        while True:
            query = input("\nEnter query (or 'exit'): ").strip()
            if query.lower() == 'exit':
                break
            tables = model.identify_tables(query)
            if tables:
                print("\nIdentified Tables:")
                for i, table in enumerate(tables, 1):
                    print(f"{i}. {table}")
            else:
                print("I am not yet trained to get relevant tables identified for this context")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()