import json
import duckdb
from datetime import datetime
from pathlib import Path
from typing import Union, List, Dict, Any

TOKEN_LIMIT = 5000

def _serialize_datetime(obj):
    """Convert datetime objects to ISO format strings for JSON serialization"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: _serialize_datetime(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_serialize_datetime(item) for item in obj]
    else:
        return obj

def _estimate_token_count(text: str) -> int:
    """Estimate token count using character-based approximation."""
    average_chars_per_token = 3
    return (len(text) + average_chars_per_token - 1) // average_chars_per_token

def _enforce_token_limit(payload: str, context: str) -> str:
    """Ensure payload stays within the token budget before returning"""
    token_estimate = _estimate_token_count(payload)
    if token_estimate <= TOKEN_LIMIT:
        return payload

    # Calculate suggested limit reduction
    try:
        current_size = len(json.loads(payload)) if payload.startswith("[") else None
    except json.JSONDecodeError:
        current_size = None
        
    suggested_limit = None
    if current_size:
        ratio = TOKEN_LIMIT / token_estimate
        suggested_limit = max(1, int(current_size * ratio * 0.8))

    suggestion_parts = [
        "The query result is too large. Please adjust your query:",
        "  • Reduce the LIMIT value" + (f" (try LIMIT {suggested_limit})" if suggested_limit else ""),
        "  • Filter rows with WHERE clauses to reduce result size",
        "  • Select only necessary columns instead of SELECT *",
        "  • Use aggregation (COUNT, SUM, AVG) instead of retrieving raw rows",
    ]

    warning = {
        "error": "Result exceeds token budget",
        "context": context,
        "estimated_tokens": token_estimate,
        "token_limit": TOKEN_LIMIT,
        "rows_returned": current_size,
        "suggested_limit": suggested_limit,
        "suggestion": "\n".join(suggestion_parts),
    }
    return json.dumps(warning, ensure_ascii=False, indent=2)

def _validate_parquet_files(parquet_files: Union[str, List[str]]) -> List[str]:
    """Validate parquet files exist and return as list."""
    if isinstance(parquet_files, str):
        parquet_files = [parquet_files]

    # Resolve paths relative to workspace root if needed, or assume absolute/relative to CWD
    # Here we assume the user passes paths that are valid from CWD
    validated_files = []
    for file_path in parquet_files:
        path = Path(file_path)
        if not path.exists():
            # Try looking in data/ folder if not found in root
            if not str(path).startswith("data/"):
                alt_path = Path("data") / path
                if alt_path.exists():
                    validated_files.append(str(alt_path))
                    continue
            
            raise FileNotFoundError(
                f"Parquet file not found: {file_path}\n"
                f"Please check the file path and ensure the file exists. "
                f"You may use 'list_tables_in_directory' to discover available parquet files."
            )
        validated_files.append(str(path))
    return validated_files

def list_tables_in_directory(directory: str = "data") -> str:
    """
    List all parquet files in the specified directory with metadata.
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        # If user passed "." but meant "data", or vice versa, try to be helpful
        if directory == "." and Path("data").exists():
             return json.dumps({"error": f"Directory not found: {directory}. Did you mean 'data'?"})
        return json.dumps({"error": f"Directory not found: {directory}"})
    
    if not dir_path.is_dir():
        return json.dumps({"error": f"Path is not a directory: {directory}"})

    files_info = []
    cwd = Path.cwd()
    
    for file_path in dir_path.glob("*.parquet"):
        file_path_str = str(file_path)
        
        try:
            conn = duckdb.connect(":memory:")
            row_count_result = conn.execute(f"SELECT COUNT(*) FROM read_parquet('{file_path_str}')").fetchone()
            if row_count_result is None:
                raise RuntimeError("Failed to read row count from parquet file")
            row_count = row_count_result[0]
            
            result = conn.execute(f"SELECT * FROM read_parquet('{file_path_str}') LIMIT 0")
            column_count = len(result.description) if result.description else 0
            conn.close()

            files_info.append(
                {
                    "filename": file_path.name,
                    "path": str(file_path),
                    "row_count": row_count,
                    "column_count": column_count,
                }
            )
        except Exception as e:
            files_info.append({
                "filename": file_path.name, 
                "path": str(file_path), 
                "error": str(e)
            })
    
    if not files_info and directory == "." and Path("data").exists():
         # If no files in root, check data/ and suggest it
         data_files = [str(f.name) for f in Path("data").glob("*.parquet")]
         if data_files:
             return json.dumps({
                 "directory": directory, 
                 "files": [], 
                 "hint": "No parquet files found in current directory. Found files in 'data/' directory. Please try list_tables_in_directory('data')."
             }, indent=2)

    return json.dumps(files_info, ensure_ascii=False, indent=2)

def get_schema(parquet_file: Union[str, List[str]]) -> str:
    """
    Get the schema (column names and types) of one or more parquet files.
    """
    try:
        parquet_files = _validate_parquet_files(parquet_file)
        
        schemas = []
        conn = duckdb.connect(":memory:")
        
        for file_path in parquet_files:
            try:
                # Get schema
                result = conn.execute(f"SELECT * FROM read_parquet('{file_path}') LIMIT 0")
                if result.description:
                    schema = [{"name": desc[0], "type": str(desc[1])} for desc in result.description]
                else:
                    schema = []

                # Get row count
                row_count_result = conn.execute(f"SELECT COUNT(*) FROM read_parquet('{file_path}')").fetchone()
                if row_count_result is None:
                    raise RuntimeError("Failed to read row count from parquet file")
                row_count = row_count_result[0]

                schema_info = {
                    "file": file_path,
                    "row_count": row_count,
                    "columns": schema,
                }
                
                # Check for special characters in column names to provide a hint
                has_special_chars = any("." in col["name"] or "-" in col["name"] for col in schema)
                if has_special_chars:
                     schema_info["note"] = "Some columns contain special characters (dots or hyphens). You MUST enclose them in double quotes in your SQL queries (e.g., \"attr.status_code\")."
                
                schemas.append(schema_info)
            except Exception as e:
                schemas.append({"file": file_path, "error": str(e)})

        if len(schemas) == 1:
            return json.dumps(schemas[0], ensure_ascii=False, indent=2)
        
        return json.dumps(schemas, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})

def query_parquet_files(parquet_files: Union[str, List[str]], query: str) -> str:
    """
    Query parquet files using SQL syntax.
    """
    table_names = set()
    try:
        parquet_files = _validate_parquet_files(parquet_files)
        
        conn = duckdb.connect(":memory:")
        
        # Register parquet files as views using filename as table name
        for file_path in parquet_files:
            base_name = Path(file_path).stem
            table_name = base_name
            counter = 1
            while table_name in table_names:
                table_name = f"{base_name}_{counter}"
                counter += 1
            table_names.add(table_name)
            # Use read_parquet for robustness
            conn.execute(f"CREATE VIEW {table_name} AS SELECT * FROM read_parquet('{file_path}')")
                
        # Execute query
        cursor = conn.execute(query)
        result = cursor.fetchall()
        
        if cursor.description is None:
            columns = []
        else:
            columns = [desc[0] for desc in cursor.description]
        
        # Convert to list of dicts
        result_dicts = [dict(zip(columns, row)) for row in result]
        
        # Serialize
        json_result = json.dumps(_serialize_datetime(result_dicts), ensure_ascii=False, indent=2)
        
        return _enforce_token_limit(json_result, query)
        
    except Exception as e:
        error_msg = str(e)
        # Provide contextual error messages similar to parquet_tools.py
        available_tables = ', '.join(table_names) if table_names else 'None'
        if "syntax error" in error_msg.lower() or "parser error" in error_msg.lower():
            return json.dumps({
                "error": f"SQL syntax error: {error_msg}",
                "hint": "Check your SQL syntax. Use 'get_schema' to verify column names. Remember to quote columns with dots like \"attr.status_code\"."
            })
        elif "catalog" in error_msg.lower() or "table" in error_msg.lower():
            return json.dumps({
                "error": f"Table/Column reference error: {error_msg}",
                "hint": f"Ensure table names match filenames (e.g., 'abnormal_logs') and column names are correct. Available tables: {available_tables}"
            })
        else:
            return json.dumps({"error": f"Query execution failed: {error_msg}"})
