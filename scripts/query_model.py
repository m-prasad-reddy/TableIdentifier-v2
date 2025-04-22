# scripts/query_model.py: Interacts with published table identifier model

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import os
import json

class TableIdentifierClient:
    """Client to query the table identifier model."""
    
    def __init__(self, model_path: str, schema_path: str):
        """Initialize with model and schema paths."""
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        with open(schema_path, 'r') as f:
            self.schema_dict = json.load(f)
        self.tables = self._get_all_tables()

    def _get_all_tables(self) -> list:
        """Get all tables from schema."""
        tables = []
        for schema in self.schema_dict['tables']:
            tables.extend(f"{schema}.{table}" for table in self.schema_dict['tables'][schema])
        return tables

    def query(self, query: str) -> list:
        """Query the model for table suggestions."""
        inputs = self.tokenizer(query, padding=True, truncation=True, return_tensors="pt")
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            max_prob, max_idx = torch.max(probs, dim=1)
            if max_prob > 0.7:
                return [self.tables[max_idx]]
            else:
                return []

if __name__ == "__main__":
    model_path = "app-config/models/table_identifier_model.pth"
    schema_path = "schema_cache/BikeStores/schema.json"
    
    if not os.path.exists(model_path) or not os.path.exists(schema_path):
        print("Model or schema not found!")
    else:
        client = TableIdentifierClient(model_path, schema_path)
        query = input("Enter query: ")
        tables = client.query(query)
        if tables:
            print(f"Suggested tables: {tables}")
        else:
            print("I am not yet trained to get relevant tables identified for this context")