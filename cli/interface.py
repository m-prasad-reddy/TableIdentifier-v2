import logging
import spacy
import os
import shutil
import json
from filelock import Timeout

class DatabaseAnalyzerCLI:
    """Command-line interface for interacting with the DatabaseAnalyzer."""
    
    def __init__(self, analyzer):
        """Initialize with analyzer instance.

        Args:
            analyzer: DatabaseAnalyzer instance.
        """
        self.logger = logging.getLogger("interface")
        self.analyzer = analyzer
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            self.logger.error(f"Failed to load spacy model: {e}")
            self.nlp = None
        self.logger.debug("Initialized DatabaseAnalyzerCLI")

    def run(self):
        """Run the main CLI loop with menu options."""
        db_name = self.analyzer.current_config.get('database', 'Database') if self.analyzer.current_config else 'Database'
        self.logger.info(f"Starting {db_name} Schema Analyzer")
        print(f"=== {db_name} Schema Analyzer ===")
        while True:
            print("\nMain Menu:")
            print("1. Connect to Database")
            print("2. Query Mode")
            print("3. Reload Configurations")
            print("4. Manage Feedback")
            print("5. Exit")
            
            choice = input("Select option: ").strip()
            
            if choice == "1":
                self._handle_connection()
                db_name = self.analyzer.current_config.get('database', 'Database') if self.analyzer.current_config else 'Database'
                print(f"\n=== {db_name} Schema Analyzer ===")
            elif choice == "2":
                self._query_mode()
            elif choice == "3":
                self._reload_configurations()
            elif choice == "4":
                self._manage_feedback()
            elif choice == "5":
                self.logger.info("Exiting application")
                print("Exiting...")
                break
            else:
                print("Invalid choice")

    def _handle_connection(self):
        """Handle database connection process."""
        config_path = input("Config path [default: app-config/database_configurations.json]: ").strip()
        if not config_path:
            config_path = "app-config/database_configurations.json"
        
        try:
            configs = self.analyzer.load_configs(config_path)
            self._select_configuration(configs)
            if self.analyzer.connect_to_database():
                self.logger.info("Successfully connected to database")
                print("Successfully connected!")
            else:
                self.logger.error("Connection failed: Unable to establish database connection")
                print("Connection failed: Unable to establish database connection")
        except Exception as e:
            self.logger.error(f"Connection failed: {str(e)}")
            print(f"Connection failed: {str(e)}")

    def _select_configuration(self, configs):
        """Select a database configuration from available options.

        Args:
            configs: Dictionary of available configurations.
        """
        print("\nAvailable Configurations:")
        for i, name in enumerate(configs.keys(), 1):
            print(f"{i}. {name}")
        print(f"{len(configs)+1}. Cancel")
        
        while True:
            choice = input("Select configuration: ").strip()
            if choice.isdigit():
                index = int(choice) - 1
                if 0 <= index < len(configs):
                    config = list(configs.values())[index]
                    self.analyzer.set_current_config(config)
                    self.logger.debug(f"Selected configuration: {config.get('database')}")
                    return
                elif index == len(configs):
                    self.logger.debug("Configuration selection cancelled")
                    return
            print("Invalid selection")

    def _validate_query(self, query):
        """Validate if a query is meaningful.

        Args:
            query: The query to validate.

        Returns:
            bool: True if the query is valid, False otherwise.
        """
        if not query or query.isspace():
            return False
            
        tokens = query.strip().split()
        if len(tokens) <= 1:
            return False
        if query.strip().isdigit():
            return False
            
        if self.nlp:
            doc = self.nlp(query.lower())
            has_noun_chunk = any(chunk for chunk in doc.noun_chunks)
            has_verb = any(token.pos_ == "VERB" for token in doc)
            
            if not (has_noun_chunk or has_verb):
                return False
                
            if len(doc) < 3 and not has_noun_chunk:
                return False
                
        return True

    def _query_mode(self):
        """Enter query mode to process natural language queries."""
        if not self.analyzer.is_connected():
            self.logger.error("Not connected to database")
            print("Not connected to database!")
            return
            
        if not self.analyzer.feedback_manager:
            self.logger.error("Feedback manager not initialized")
            print("Feedback manager not initialized. Please reconnect to the database.")
            return
            
        try:
            example_queries = self.analyzer.feedback_manager.get_top_queries(3)
            if example_queries:
                print("\nExample Queries:")
                for i, (query, count) in enumerate(example_queries, 1):
                    print(f"{i}. {query} (used {count} times)")
            else:
                print("\nNo example queries available. Try these formats:")
                print("1. Show me all stores with store names")
                print("2. List all products with prices")
                print("3. Show customers from a specific city")
        except Exception as e:
            self.logger.error(f"Error loading example queries: {str(e)}")
            print(f"Error loading example queries: {str(e)}")
            
        while True:
            query = input("\nEnter query (or 'back'): ").strip()
            if query.lower() == 'back':
                self.logger.debug("Exiting query mode")
                return
                
            if not self._validate_query(query):
                self.logger.warning(f"Invalid query: {query}")
                print("Please enter a meaningful query (e.g., 'show me all stores with store names').")
                print("Avoid single words, numbers, or vague phrases.")
                continue
                
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    results, confidence = self.analyzer.process_query(query)
                    if results is None:
                        self.logger.error("Unable to process query")
                        print("Unable to process query. Please try again or reconnect.")
                        continue
                        
                    if confidence and results:
                        self.logger.info(f"Suggested tables for query '{query}': {results}")
                        print("\nSuggested Tables:")
                        for i, table in enumerate(results[:5], 1):
                            print(f"{i}. {table}")
                        self._handle_feedback(query, results)
                        break
                    else:
                        self.logger.warning(f"Low confidence for query '{query}'")
                        print("\nLow confidence. Please select tables manually:")
                        self._manual_table_selection(query)
                        break
                except Timeout:
                    self.logger.warning(f"Feedback lock timeout on attempt {attempt + 1}/{max_retries}")
                    if attempt == max_retries - 1:
                        self.logger.error("Feedback lock timeout after max retries")
                        print("Feedback system is busy. Please select tables manually.")
                        self._manual_table_selection(query)
                        break
                except Exception as e:
                    self.logger.error(f"Error processing query: {str(e)}")
                    print(f"Error processing query: {str(e)}")
                    break

    def _handle_feedback(self, query, results):
        """Handle user feedback for suggested tables.

        Args:
            query: The query.
            results: Suggested tables.
        """
        while True:
            feedback = input("\nCorrect? (Y/N): ").strip().lower()
            if feedback in ('y', 'n'):
                break
            print("Please enter 'Y' or 'N'.")
        
        if feedback == 'y':
            self.logger.info(f"Confirmed tables for query '{query}': {results}")
            self.analyzer.confirm_tables(query, results)
        elif feedback == 'n':
            self.logger.info(f"User rejected tables for query '{query}': {results}")
            correct_tables = self._get_manual_tables()
            if correct_tables:
                self.logger.info(f"Updated feedback with tables: {correct_tables}")
                self.analyzer.update_feedback(query, correct_tables)

    def _get_manual_tables(self):
        """Get manual table selections from the user.

        Returns:
            List of selected tables.
        """
        print("Available Tables:")
        tables = self.analyzer.get_all_tables()
        for i, table in enumerate(tables, 1):
            print(f"{i}. {table}")
            
        selection = input("Enter table numbers or names (comma-separated, e.g., '6' or 'sales.stores'): ").strip()
        if not selection:
            self.logger.debug("No manual tables selected")
            return []
            
        selected = []
        items = [s.strip() for s in selection.split(',')]
        
        for item in items:
            if item.isdigit():
                try:
                    index = int(item) - 1
                    if 0 <= index < len(tables):
                        selected.append(tables[index])
                except (IndexError, ValueError):
                    continue
            elif '.' in item:
                schema, table_name = item.split('.', 1)
                if any(t.lower() == item.lower() for t in tables):
                    selected.append(item)
        
        if not selected:
            self.logger.warning("Invalid manual table selection")
            print("Invalid selection, please try again")
        else:
            self.logger.debug(f"Selected manual tables: {selected}")
        return selected

    def _manual_table_selection(self, query):
        """Handle manual table selection for a query.

        Args:
            query: The query.
        """
        selected_tables = self._get_manual_tables()
        if selected_tables:
            self.logger.info(f"Manually selected tables for query '{query}': {selected_tables}")
            self.analyzer.update_feedback(query, selected_tables)

    def _reload_configurations(self):
        """Reload all configurations and caches."""
        try:
            if self.analyzer.reload_all_configurations():
                self.logger.info("Successfully reloaded configurations")
                print("Successfully reloaded configurations")
            else:
                self.logger.error("Reload failed: Unable to reload configurations")
                print("Reload failed: Unable to reload configurations")
        except Exception as e:
            self.logger.error(f"Reload failed: {str(e)}")
            print(f"Reload failed: {str(e)}")

    def _manage_feedback(self):
        """Manage feedback operations (export, import, clear)."""
        if not self.analyzer.is_connected():
            self.logger.error("Not connected to database for feedback management")
            print("Please connect to a database to manage feedback.")
            return
            
        if not self.analyzer.feedback_manager:
            self.logger.error("Feedback manager not initialized")
            print("Feedback manager not initialized. Please reconnect to the database.")
            return
            
        print("\nFeedback Management:")
        print("1. Export feedback")
        print("2. Import feedback")
        print("3. Clear local feedback")
        choice = input("Select option: ").strip()
        
        if choice == "1":
            self._export_feedback()
        elif choice == "2":
            self._import_feedback()
        elif choice == "3":
            try:
                self.analyzer.clear_feedback()
                self.logger.info("Feedback cleared")
            except Exception as e:
                self.logger.error(f"Error clearing feedback: {str(e)}")
                print(f"Error clearing feedback: {str(e)}")
        else:
            print("Invalid choice")

    def _export_feedback(self):
        """Export feedback data to a specified directory."""
        export_dir = input("Enter export directory path [default: feedback_cache/export]: ").strip()
        if not export_dir:
            export_dir = os.path.join("feedback_cache", "export")
            
        try:
            os.makedirs(export_dir, exist_ok=True)
            feedback_dir = self.analyzer.feedback_manager.feedback_dir
            copied = False
            for fname in os.listdir(feedback_dir):
                if fname.endswith("_meta.json"):
                    src = os.path.join(feedback_dir, fname)
                    with open(src) as f:
                        meta = json.load(f)
                    if 'query' not in meta or 'tables' not in meta or 'timestamp' not in meta:
                        self.logger.warning(f"Skipping invalid feedback file: {fname}")
                        print(f"Skipping invalid feedback file: {fname}")
                        continue
                    dst = os.path.join(export_dir, fname)
                    shutil.copy2(src, dst)
                    emb_fname = fname.replace("_meta.json", "_emb.npy")
                    emb_src = os.path.join(feedback_dir, emb_fname)
                    if os.path.exists(emb_src):
                        shutil.copy2(emb_src, os.path.join(export_dir, emb_fname))
                    copied = True
            if copied:
                self.logger.info(f"Feedback exported to {export_dir}")
                print(f"Feedback exported to {export_dir}")
            else:
                self.logger.info("No valid feedback files to export")
                print("No valid feedback files to export")
        except Exception as e:
            self.logger.error(f"Error exporting feedback: {str(e)}")
            print(f"Error exporting feedback: {str(e)}")

    def _import_feedback(self):
        """Import feedback data from a specified directory."""
        import_dir = input("Enter import directory path: ").strip()
        if not import_dir or not os.path.exists(import_dir):
            self.logger.error("Invalid or non-existent import directory")
            print("Invalid or non-existent directory")
            return
            
        try:
            feedback_dir = self.analyzer.feedback_manager.feedback_dir
            os.makedirs(feedback_dir, exist_ok=True)
            copied = False
            for fname in os.listdir(import_dir):
                if fname.endswith("_meta.json"):
                    src = os.path.join(import_dir, fname)
                    with open(src) as f:
                        meta = json.load(f)
                    if 'query' not in meta or 'tables' not in meta or 'timestamp' not in meta:
                        self.logger.warning(f"Skipping invalid feedback file: {fname}")
                        print(f"Skipping invalid feedback file: {fname}")
                        continue
                    dst = os.path.join(feedback_dir, fname)
                    shutil.copy2(src, dst)
                    emb_fname = fname.replace("_meta.json", "_emb.npy")
                    emb_src = os.path.join(import_dir, emb_fname)
                    if os.path.exists(emb_src):
                        shutil.copy2(emb_src, os.path.join(feedback_dir, emb_fname))
                    copied = True
            if copied:
                self.analyzer.feedback_manager._load_feedback_cache()
                self.logger.info(f"Feedback imported from {import_dir}")
                print(f"Feedback imported from {import_dir}")
            else:
                self.logger.info("No valid feedback files to import")
                print("No valid feedback files to import")
        except Exception as e:
            self.logger.error(f"Error importing feedback: {str(e)}")
            print(f"Error importing feedback: {str(e)}")