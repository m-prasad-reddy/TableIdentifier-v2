# TableIdentifier-Working-Version-2 Documentation

## Overview

**TableIdentifier-Working-Version-2** is a Python-based application designed to connect to relational databases, analyze their schemas, and process natural language queries to identify relevant tables. The application is built to support multiple database types (e.g., SQL Server, PostgreSQL) and is extensible to handle various database schemas, with dynamic pattern learning and CSV-based training for enhanced table identification accuracy. It provides a command-line interface (CLI) for user interaction, enabling database connections, query processing, feedback management, and configuration reloading.

The application is implemented in a modular structure, with core functionality distributed across three primary files:
- `main.py`: Orchestrates the application workflow.
- `schema/manager.py`: Manages schema metadata extraction and caching.
- `cli/interface.py`: Provides the interactive CLI.

Additional modules handle specific tasks such as database connections, configuration management, pattern recognition, feedback storage, table identification, name matching, NLP processing, and query coordination. The codebase is designed to be extensible, allowing the addition of new databases and database types. This documentation addresses issues identified in logs, including `FileLock` hangs, `42S22` schema mtime errors, and CSV parsing errors, with fixes applied to ensure robust operation.

## Purpose

The primary purpose of **TableIdentifier-Working-Version-2** is to:
- **Connect to Multiple Databases**: Establish connections to various relational databases using `pyodbc`, supporting SQL Server, PostgreSQL, and other database types, with configurations loaded from a JSON file.
- **Analyze Schemas**: Extract and cache schema metadata (tables, columns, indexes, foreign keys, views) for efficient query processing across different database types.
- **Process Natural Language Queries**: Identify relevant tables from user queries using NLP techniques (`spacy`, `sentence_transformers`) and feedback-driven learning, adaptable to diverse schemas.
- **Provide a CLI**: Offer an interactive interface for users to connect to databases, enter queries, manage feedback, and reload configurations.
- **Support Feedback**: Store and utilize user feedback to improve table identification accuracy, with thread-safe operations and export/import capabilities.
- **Ensure Extensibility**: Enable easy integration of new databases and database types through modular design and configuration-driven connectivity.

## Architecture

The application is structured around three core modules and eight supporting modules, organized within the `EntityResolver` directory:

### Core Modules
1. **DatabaseAnalyzer** (`main.py`):
   - Orchestrates the application, managing database connections, schema initialization, query processing, and feedback integration.
   - Coordinates interactions between all modules.

2. **SchemaManager** (`schema/manager.py`):
   - Extracts and caches schema metadata, supporting multiple database types (SQL Server, PostgreSQL, generic).
   - Validates schema consistency and optimizes access through caching.

3. **DatabaseAnalyzerCLI** (`cli/interface.py`):
   - Provides a command-line interface for user operations, including database selection, query entry, feedback management, and configuration reloading.
   - Validates queries using `spacy` for meaningfulness.

### Supporting Modules
1. **DatabaseConnection**:
   - Manages `pyodbc` connections to relational databases, supporting multiple database types.
   - Handles connection establishment and closure.

2. **DBConfigManager**:
   - Loads database configurations from a JSON file, enabling connectivity to various databases.
   - Supports extensibility for new database configurations.

3. **PatternManager**:
   - Manages dynamic patterns for table identification, derived from schema metadata.
   - Adapts patterns to different database schemas.

4. **FeedbackManager**:
   - Stores and retrieves user feedback on query-table mappings, using `FileLock` for thread safety.
   - Supports feedback export/import and similarity-based retrieval.

5. **TableIdentifier**:
   - Identifies relevant tables from queries, leveraging CSV training data and feedback.
   - Updates identification weights based on user feedback.

6. **NameMatchManager**:
   - Matches query terms to schema entities (tables, columns) using similarity metrics.
   - Enhances table identification accuracy.

7. **NLPPipeline**:
   - Processes queries using NLP techniques (`spacy`, `sentence_transformers`) to extract meaningful entities.
   - Adapts to various query structures and database schemas.

8. **QueryProcessor**:
   - Coordinates query processing, integrating NLP, table identification, and schema data.
   - Ensures accurate table suggestions across database types.

## Module Functionalities

### 1. DatabaseAnalyzer (`main.py`)
**Path**: `C:\Users\User1\Pythonworks\TableIdentifier-2\main.py`
**Purpose**: Central orchestrator for database connections, schema management, query processing, and feedback integration.
**Key Methods**:
- `__init__`: Initializes logging, connection manager, configuration manager, and component placeholders.
- `run`: Launches the CLI, manages shutdown, saves models, and closes connections.
- `load_configs`: Loads database configurations from `app-config/database_configurations.json`.
- `set_current_config`: Sets the active database configuration.
- `connect_to_database`: Establishes a database connection and initializes supporting modules.
- `_initialize_managers`: Initializes schema, pattern, feedback, NLP, and query processing components.
- `reload_all_configurations`: Rebuilds schema and reinitializes modules.
- `process_query`: Processes natural language queries to identify tables, handling `FileLock` timeouts.
- `validate_tables_exist`: Validates identified tables against the schema.
- `generate_ddl`: Generates DDL statements for specified tables.
- `close_connection`: Closes the database connection.
- `is_connected`: Checks connection status.
- `get_all_tables`: Retrieves all schema tables.
- `confirm_tables`/`update_feedback`: Stores feedback for query-table mappings.
- `clear_feedback`: Clears stored feedback.

**Fixes Applied**:
- Handles `NoneType` errors by resetting modules on initialization failure.
- Catches CSV parsing errors in `TableIdentifier` initialization, allowing partial functionality.
- Handles `FileLock` timeouts in `process_query` with fallback to empty results.

### 2. SchemaManager (`schema/manager.py`)
**Path**: `C:\Users\User1\Pythonworks\TableIdentifier-2\schema\manager.py`
**Purpose**: Extracts and caches schema metadata, supporting multiple database types.
**Key Methods**:
- `__init__`: Initializes logging, cache directory (`schema_cache/<db_name>`), and database type.
- `set_db_type`: Detects database type (SQL Server, PostgreSQL, generic) using `@@VERSION`.
- `needs_refresh`: Checks if the schema cache needs refreshing based on modification timestamps.
- `_get_schema_mtime`: Retrieves the latest schema modification timestamp, tailored to database type.
- `build_data_dict`: Builds a schema dictionary (tables, columns, indexes, foreign keys, views).
- `_validate_schema`: Validates schema consistency (e.g., checks for missing columns or invalid foreign keys).
- `load_from_cache`: Loads schema from cache if valid.

**Fixes Applied**:
- Corrected SQL Server query in `_get_schema_mtime` to use `create_date` and `modify_date` from `sys.tables` and `sys.columns`, resolving `42S22` errors.
- Uses separate subqueries with LEFT JOINs in `build_data_dict` to avoid `21000` errors.

### 3. DatabaseAnalyzerCLI (`cli/interface.py`)
**Path**: `C:\Users\User1\Pythonworks\TableIdentifier-2\cli\interface.py`
**Purpose**: Provides an interactive CLI for user operations across multiple databases.
**Key Methods**:
- `__init__`: Initializes logging and loads `spacy` model (`en_core_web_sm`).
- `run`: Runs the main CLI loop with menu options (connect, query, reload, feedback, exit).
- `_handle_connection`: Manages database connection process, supporting various database types.
- `_select_configuration`: Allows users to select a database configuration.
- `_validate_query`: Validates query meaningfulness using `spacy`.
- `_query_mode`: Processes natural language queries, handles feedback, and retries on `FileLock` timeouts.
- `_handle_feedback`: Collects user feedback on suggested tables.
- `_get_manual_tables`: Allows manual table selection for low-confidence queries.
- `_manual_table_selection`: Handles manual table input.
- `_reload_configurations`: Reloads configurations via `DatabaseAnalyzer`.
- `_manage_feedback`: Manages feedback export/import/clear operations.
- `_export_feedback`: Exports feedback to a specified directory.
- `_import_feedback`: Imports feedback from a directory.

**Fixes Applied**:
- Added retry mechanism (3 attempts, 5-second timeout) for `FileLock` in `_query_mode` to handle hangs.
- Checks for `FeedbackManager` initialization to prevent `NoneType` errors.
- Gracefully handles `spacy` model loading failures.

### 4. DatabaseConnection
**Purpose**: Manages `pyodbc` connections to relational databases, supporting SQL Server, PostgreSQL, and other types.
**Key Methods** (Assumed):
- `connect(config)`: Establishes a connection using the provided configuration.
- `close()`: Closes the active connection.
- `is_connected()`: Checks if a connection is active.
- `connection`: Property to access the active connection object.

**Role**: Provides a unified interface for database connectivity, enabling the application to work with multiple database types through configuration-driven settings.

### 5. DBConfigManager
**Purpose**: Loads and manages database configurations from a JSON file, supporting extensibility for new databases.
**Key Methods** (Assumed):
- `__init__`: Initializes the configuration manager.
- `load_configs(config_path)`: Loads configurations from a JSON file, returning a dictionary of database settings.

**Role**: Enables the application to connect to various databases by reading configurations (e.g., server, database name, credentials) from `app-config/database_configurations.json`.

### 6. PatternManager
**Purpose**: Manages dynamic patterns for table identification, derived from schema metadata.
**Key Methods** (Assumed):
- `__init__(schema_dict)`: Initializes with schema data to build patterns.
- `load_patterns()`: Loads predefined patterns from `app-config/global_patterns.json`.
- `match_pattern(query)`: Matches query terms to patterns for table identification.

**Role**: Enhances table identification by recognizing schema-specific patterns, adaptable to different database schemas.

### 7. FeedbackManager
**Purpose**: Stores and retrieves user feedback on query-table mappings, using `FileLock` for thread-safe operations.
**Key Methods** (Assumed):
- `__init__(db_name)`: Initializes feedback storage in `feedback_cache/<db_name>`.
- `store_feedback(query, tables, schema_dict)`: Saves feedback for a query-table mapping.
- `get_similar_feedback(query, threshold)`: Retrieves feedback for similar queries using embeddings.
- `get_top_queries(limit)`: Returns the most frequent queries.
- `_load_feedback_cache()`: Loads cached feedback data.
- `clear_feedback()`: Clears all feedback data.

**Fixes Applied**:
- Persistent `FileLock` hangs in `get_similar_feedback` addressed with retries and timeouts in `DatabaseAnalyzerCLI`.

### 8. TableIdentifier
**Purpose**: Identifies relevant tables from queries, leveraging CSV training data and feedback.
**Key Methods** (Assumed):
- `__init__(schema_dict, feedback_manager, pattern_manager)`: Initializes with schema, feedback, and patterns.
- `identify_tables(query)`: Returns a list of tables and confidence score for a query.
- `save_name_matches()`: Saves name matching data.
- `save_model(model_path)`: Saves the trained model.
- `update_weights_from_feedback(query, tables)`: Updates identification weights based on feedback.

**Fixes Applied**:
- Handles CSV parsing errors (`Expected 13 fields in line 3, saw 16`) by skipping initialization, logging the issue.

### 9. NameMatchManager
**Purpose**: Matches query terms to schema entities (tables, columns) using similarity metrics.
**Key Methods** (Assumed):
- `__init__(db_name)`: Initializes with database-specific settings.
- `match_names(query, schema_dict)`: Returns matched tables/columns based on query terms.

**Role**: Improves table identification by aligning query terms with schema entities, supporting diverse schemas.

### 10. NLPPipeline
**Purpose**: Processes queries using NLP techniques to extract meaningful entities.
**Key Methods** (Assumed):
- `__init__(pattern_manager, db_name)`: Initializes with patterns and database context.
- `process_query(query)`: Extracts entities and embeddings from queries.

**Role**: Enables robust query processing by leveraging `spacy` and `sentence_transformers`, adaptable to various query structures.

### 11. QueryProcessor
**Purpose**: Coordinates query processing, integrating NLP, table identification, and schema data.
**Key Methods** (Assumed):
- `__init__(connection_manager, schema_dict, nlp_pipeline, table_identifier, name_matcher, pattern_manager, db_name)`: Initializes with all necessary components.
- `process_query(query)`: Processes a query, returning identified tables and confidence.

**Role**: Ensures accurate table suggestions by orchestrating NLP, matching, and identification processes.

## Dependencies

- **Python**: 3.8+ (based on your virtual environment).
- **pyodbc**: For database connections (SQL Server, PostgreSQL).
- **spacy**: For NLP query validation (`en_core_web_sm` model).
- **sentence_transformers**: For query embeddings (`all-distilroberta-v1`).
- **filelock**: For thread-safe feedback management.
- **json, shutil, os**: Standard library modules for file operations.
- **logging**: For application logging (`app-config/logging_config.ini`).
- **collections.defaultdict**: For schema dictionary management.
- **typing**: For type hints.

**Installation**:
```bash
pip install pyodbc spacy sentence-transformers filelock
python -m spacy download en_core_web_sm
```

## Project Structure

```
C:\Users\User1\Pythonworks\TableIdentifier-2\
├── main.py                           # DatabaseAnalyzer implementation
├── schema/
│   ├── manager.py                    # SchemaManager implementation
├── cli/
│   ├── interface.py                  # DatabaseAnalyzerCLI implementation
├── database/
│   ├── connection.py                 # DatabaseConnection implementation
├── config/
│   ├── config_manager.py             # DBConfigManager implementation
│   ├── patterns.py                   # PatternManager implementation
├── feedback/
│   ├── feedback_manager.py           # FeedbackManager implementation
├── analysis/
│   ├── table_identifier.py           # TableIdentifier implementation
│   ├── name_match_manager.py         # NameMatchManager implementation
│   ├── processor.py                  # NLPPipeline implementation
├── nlp/
│   ├── query_processor.py            # QueryProcessor implementation
├── app-config/
│   ├── database_configurations.json  # Database configurations
│   ├── logging_config.ini            # Logging configuration
│   ├── global_patterns.json          # Pattern definitions
├── schema_cache/
│   ├── <db_name>/
│   │   ├── schema.json               # Cached schema
├── feedback_cache/
│   ├── <db_name>/                    # Feedback files (_meta.json, _emb.npy)
│   ├── export/                       # Exported feedback
├── models/
│   ├── <db_name>_model.json          # Saved table identifier model
├── logs/
│   ├── app.log                       # Application logs
```

## Usage Instructions

1. **Setup Environment**:
   - Activate the virtual environment:
     ```bash
     C:\Users\User1\Pythonworks\Text2SQl\venv\Scripts\activate
     ```
   - Ensure dependencies are installed.

2. **Run the Application**:
   - Execute `main.py`:
     ```bash
     python C:\Users\User1\Pythonworks\TableIdentifier-2\main.py
     ```

3. **CLI Menu**:
   - **Connect to Database**: Select option 1, choose a database configuration (e.g., `BIKES_DB`, `ADWORKS_DB`), and confirm the configuration path (`app-config/database_configurations.json`).
   - **Query Mode**: Select option 2, enter a query (e.g., "Show me all stores with store names"), and confirm or correct suggested tables.
   - **Reload Configurations**: Select option 3 to refresh schema and modules.
   - **Manage Feedback**: Select option 4 to export/import/clear feedback.
   - **Exit**: Select option 5 to shut down.

4. **Example Query**:
   - Input: "Show me all stores with store names"
   - Expected Output: Suggested tables (e.g., `sales.stores`), with option to confirm or select manually.

## Known Issues and Fixes

1. **FileLock Hang**:
   - **Issue**: `FeedbackManager.get_similar_feedback` hangs due to `FileLock` contention (`feedback.lock`).
   - **Fix**: Implemented 3 retry attempts with a 5-second timeout in `cli/interface.py` (`_query_mode`). Falls back to manual table selection if retries fail.
   - **Status**: Fixed in `cli/interface.py`.

2. **Schema mtime Error (`42S22`)**:
   - **Issue**: `_get_schema_mtime` in `schema/manager.py` uses incorrect column names, causing SQL Server errors and forcing schema rebuilds.
   - **Fix**: Updated SQL Server query to use `create_date` and `modify_date` from `sys.tables` and `sys.columns`.
   - **Status**: Fixed in `schema/manager.py`.

3. **CSV Parsing Error**:
   - **Issue**: `TableIdentifier` fails to parse a CSV file (`Expected 13 fields in line 3, saw 16`), likely in `app-config/`.
   - **Fix**: Added error handling in `main.py` (`_initialize_managers`) to skip `TableIdentifier` initialization, logging the error.
   - **Status**: Temporary fix implemented. Requires CSV file details for a permanent solution.

4. **NoneType Error**:
   - **Issue**: `FeedbackManager` was `None` when calling `get_top_queries` in `cli/interface.py`.
   - **Fix**: Added initialization checks in `cli/interface.py`.
   - **Status**: Resolved.

## Debugging Tips

- **Logs**: Check `app.log` and `Console.txt` in `C:\Users\User1\Pythonworks\TableIdentifier-2\logs\` for errors.
- **CSV Issue**: Inspect the CSV file used by `TableIdentifier`. Verify line 3 has 13 fields. Share the file or its structure for a specific fix.
- **FileLock**: If hangs persist, reduce the timeout (e.g., 2 seconds) in `cli/interface.py` or disable `FileLock` temporarily in `FeedbackManager`.
- **Schema Cache**: If schema errors occur, delete `schema_cache/<db_name>/schema.json` to force a rebuild.
- **Database Types**: Test with PostgreSQL or other database types by updating `app-config/database_configurations.json` and verifying `DatabaseConnection` compatibility.

## Future Improvements

- **Database Extensibility**: Enhance `DatabaseConnection` and `SchemaManager` to support additional database types (e.g., MySQL, Oracle) with minimal code changes.
- **CSV Validation**: Implement robust CSV parsing with field validation in `TableIdentifier`.
- **Feedback Optimization**: Reduce `FileLock` contention by batching feedback writes or using a database backend.
- **Query Performance**: Cache query embeddings in `NLPPipeline` to speed up processing.
- **GUI Interface**: Develop a graphical interface to complement the CLI, improving user accessibility.

## Sharing with Collaborators

- **Provide Files**:
  - Share `main.py`, `schema/manager.py`, `cli/interface.py`, and this documentation (`TableIdentifier_Documentation.md`).
  - Include `app.log` and `Console.txt` for context.
  - Optionally provide `database/connection.py`, `config/config_manager.py`, etc., if collaborators need the full codebase.
- **Instructions**:
  - Direct collaborators to save files in `C:\Users\User1\Pythonworks\TableIdentifier-2\` with the appropriate directory structure.
  - Suggest running in the virtual environment and checking logs.
  - Highlight the "Known Issues and Fixes" section for debugging focus.
- **External Hosting**:
  - **GitHub Gist**: Create a Gist at `https://gist.github.com/` with `main.py`, `schema/manager.py`, `cli/interface.py`, and `TableIdentifier_Documentation.md`. Make it public and share the link.
  - **Google Drive**: Upload files to a shared Google Drive folder and provide the link.
  - Note that the application supports multiple database types, with current testing focused on SQL Server (BikeStores).

## Contact for Support

For further assistance, share:
- Updated `app.log` and `Console.txt` after testing.
- CSV file details (path, structure, or content) used by `TableIdentifier`.
- Any new errors or specific issues encountered.
- Details on additional database types tested (e.g., PostgreSQL configurations).

This documentation provides a clear, comprehensive guide for collaborators to understand, debug, and extend **TableIdentifier-Working-Version-2** for multiple databases and database types.
