# config/trainer.py: Manages CSV-based training data

import os
import pandas as pd
import json
import logging
import logging.config
from typing import Dict

class Trainer:
    """Manages training data from CSV/Excel for table identification."""
    
    def __init__(self, db_name: str, schema_dict: Dict):
        """Initialize with database name and schema."""
        logging_config_path = "app-config/logging_config.ini"
        if os.path.exists(logging_config_path):
            try:
                logging.config.fileConfig(logging_config_path, disable_existing_loggers=False)
            except Exception as e:
                print(f"Error loading logging config: {e}")
        
        self.logger = logging.getLogger("trainer")
        self.db_name = db_name
        self.schema_dict = schema_dict
        self.trainer_path = os.path.join("app-config", db_name, "db_config_trainer.csv")
        self.training_data = None
        self.logger.debug(f"Initialized Trainer for {db_name}")

    def load_training_data(self):
        """Load or create training data from CSV."""
        if os.path.exists(self.trainer_path):
            try:
                self.training_data = pd.read_csv(self.trainer_path)
                self.logger.debug(f"Loaded training data from {self.trainer_path}")
            except Exception as e:
                self.logger.error(f"Error loading training data: {e}")
                self._create_template()
        else:
            self._create_template()

    def _create_template(self):
        """Create a template CSV if none exists."""
        columns = [
            "DB_Config", "Schema", "Table_Name", "Primary_Keys", "Foreign_Keys",
            "Associated_Tables", "Associated_Views", "Description", "Columns_List"
        ]
        template_data = []
        for schema in self.schema_dict["tables"]:
            for table in self.schema_dict["tables"][schema]:
                columns_list = ",".join(self.schema_dict["columns"][schema][table].keys())
                template_data.append({
                    "DB_Config": self.db_name,
                    "Schema": schema,
                    "Table_Name": table,
                    "Primary_Keys": "",
                    "Foreign_Keys": "",
                    "Associated_Tables": "",
                    "Associated_Views": "",
                    "Description": f"Description of {schema}.{table}",
                    "Columns_List": columns_list
                })
        
        df = pd.DataFrame(template_data, columns=columns)
        os.makedirs(os.path.dirname(self.trainer_path), exist_ok=True)
        df.to_csv(self.trainer_path, index=False)
        self.training_data = df
        self.logger.debug(f"Created template CSV at {self.trainer_path}")

    def update_configs(self, pattern_manager, name_matcher, feedback_manager):
        """Update configs based on training data."""
        if self.training_data is None:
            self.logger.warning("No training data loaded")
            return

        # Update name matches
        name_matches = {}
        for _, row in self.training_data.iterrows():
            columns = row["Columns_List"].split(",")
            for col in columns:
                if col:
                    name_matches[col.lower()] = [col.lower()]
        name_matcher.default_matches.update(name_matches)
        name_matcher.save_to_default()

        # Update patterns
        patterns = pattern_manager.get_patterns()
        for _, row in self.training_data.iterrows():
            desc = row["Description"].lower()
            table = f"{row['Schema']}.{row['Table_Name']}"
            patterns[desc] = {table: 1.0}
        with open("app-config/global_patterns.json", 'w') as f:
            json.dump(patterns, f, indent=2)
        pattern_manager.pattern_weights = patterns
        pattern_manager.logger.debug("Updated patterns from training data")

        # Update feedback
        for _, row in self.training_data.iterrows():
            desc = row["Description"]
            table = f"{row['Schema']}.{row['Table_Name']}"
            feedback_manager.store_feedback(desc, [table], self.schema_dict)
        self.logger.debug("Updated configs from training data")