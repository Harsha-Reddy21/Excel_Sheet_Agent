from langchain.tools import BaseTool, StructuredTool, tool
from langchain.pydantic_v1 import BaseModel, Field
from typing import Dict, List, Optional, Union, Any
import pandas as pd
from excel_processor import (
    read_excel_file, filter_data, aggregate_data, sort_data,
    create_pivot_table, merge_worksheets, extract_date_range,
    validate_data, map_column_names, detect_data_types, infer_column_purpose
)
import json
import tempfile
import os

# Global state to store loaded Excel data
excel_state = {
    "file_path": None,
    "sheets": {},
    "active_sheet": None,
    "active_data": None,
    "results": {},
    "column_purposes": {},  # Store inferred column purposes
    "data_types": {}        # Store detected data types
}

@tool
def load_excel_file(file_path: str) -> str:
    """
    Load an Excel file and store its contents in memory.
    
    Args:
        file_path: Path to the Excel file
        
    Returns:
        Summary of loaded sheets
    """
    try:
        excel_state["file_path"] = file_path
        excel_state["sheets"] = read_excel_file(file_path)
        
        # Set the first sheet as active by default
        first_sheet = next(iter(excel_state["sheets"]))
        excel_state["active_sheet"] = first_sheet
        excel_state["active_data"] = excel_state["sheets"][first_sheet]
        
        # Detect data types and infer column purposes for each sheet
        for sheet_name, df in excel_state["sheets"].items():
            excel_state["data_types"][sheet_name] = detect_data_types(df)
            excel_state["column_purposes"][sheet_name] = infer_column_purpose(df)
        
        sheet_info = []
        for name, df in excel_state["sheets"].items():
            sheet_info.append({
                "name": name,
                "rows": len(df),
                "columns": list(df.columns)[:10] + (["..."] if len(df.columns) > 10 else []),
                "column_purposes": excel_state["column_purposes"][name]
            })
        
        return json.dumps(sheet_info, indent=2)
    except Exception as e:
        return f"Error loading Excel file: {str(e)}"

@tool
def get_file_status() -> str:
    """
    Get the status of the currently loaded Excel file.
    Use this to check if a file is loaded and what file it is.
    
    Returns:
        Status information about the loaded file
    """
    if excel_state["file_path"] is None:
        return "No Excel file is currently loaded."
    
    status = {
        "file_path": excel_state["file_path"],
        "active_sheet": excel_state["active_sheet"],
        "sheets": {name: len(df) for name, df in excel_state["sheets"].items()},
        "total_rows": sum(len(df) for df in excel_state["sheets"].values())
    }
    
    return json.dumps(status, indent=2)

@tool
def switch_sheet(sheet_name: str) -> str:
    """
    Switch the active sheet.
    
    Args:
        sheet_name: Name of the sheet to switch to
        
    Returns:
        Confirmation message
    """
    if not excel_state["sheets"]:
        return "No Excel file loaded. Please load a file first."
    
    if sheet_name not in excel_state["sheets"]:
        return f"Sheet '{sheet_name}' not found. Available sheets: {list(excel_state['sheets'].keys())}"
    
    excel_state["active_sheet"] = sheet_name
    excel_state["active_data"] = excel_state["sheets"][sheet_name]
    
    return f"Switched to sheet '{sheet_name}'. {len(excel_state['active_data'])} rows, {len(excel_state['active_data'].columns)} columns."

@tool
def get_column_info(sheet_name: Optional[str] = None) -> str:
    """
    Get information about columns in the active or specified sheet.
    
    Args:
        sheet_name: Optional name of the sheet to get column info for
        
    Returns:
        Column information as JSON string
    """
    if not excel_state["sheets"]:
        return "No Excel file loaded. Please load a file first."
    
    target_sheet = sheet_name if sheet_name else excel_state["active_sheet"]
    
    if target_sheet not in excel_state["sheets"]:
        return f"Sheet '{target_sheet}' not found. Available sheets: {list(excel_state['sheets'].keys())}"
    
    df = excel_state["sheets"][target_sheet]
    
    column_info = {}
    for col in df.columns:
        column_info[col] = {
            "dtype": str(df[col].dtype),
            "unique_values": df[col].nunique(),
            "sample": df[col].head(3).tolist() if len(df) > 0 else [],
            "purpose": excel_state["column_purposes"].get(target_sheet, {}).get(col, "unknown"),
            "missing_values": df[col].isna().sum()
        }
    
    return json.dumps(column_info, indent=2, default=str)

@tool
def filter_sheet_data(conditions: Dict[str, Any]) -> str:
    """
    Filter data in the active sheet based on conditions.
    
    Args:
        conditions: Dictionary of column-condition pairs
        
    Returns:
        Summary of filtered data
    """
    if not excel_state["active_data"] is not None:
        return "No active sheet. Please load a file and select a sheet first."
    
    try:
        filtered_df = filter_data(excel_state["active_data"], conditions)
        
        # Store the result
        result_id = f"filter_result_{len(excel_state['results'])}"
        excel_state["results"][result_id] = filtered_df
        excel_state["active_data"] = filtered_df  # Update active data
        
        return f"Filtered data: {len(filtered_df)} rows. Result ID: {result_id}"
    except Exception as e:
        return f"Error filtering data: {str(e)}"

@tool
def aggregate_sheet_data(group_by: List[str], aggregations: Dict[str, str]) -> str:
    """
    Aggregate data in the active sheet.
    
    Args:
        group_by: List of columns to group by
        aggregations: Dictionary mapping columns to aggregation functions
        
    Returns:
        Summary of aggregated data
    """
    if excel_state["active_data"] is None:
        return "No active sheet. Please load a file and select a sheet first."
    
    try:
        aggregated_df = aggregate_data(excel_state["active_data"], group_by, aggregations)
        
        # Store the result
        result_id = f"aggregate_result_{len(excel_state['results'])}"
        excel_state["results"][result_id] = aggregated_df
        excel_state["active_data"] = aggregated_df  # Update active data
        
        return f"Aggregated data: {len(aggregated_df)} rows. Result ID: {result_id}"
    except Exception as e:
        return f"Error aggregating data: {str(e)}"

@tool
def sort_sheet_data(sort_by: List[str], ascending: Union[bool, List[bool]] = True) -> str:
    """
    Sort data in the active sheet.
    
    Args:
        sort_by: List of columns to sort by
        ascending: Whether to sort in ascending order
        
    Returns:
        Confirmation message
    """
    if excel_state["active_data"] is None:
        return "No active sheet. Please load a file and select a sheet first."
    
    try:
        sorted_df = sort_data(excel_state["active_data"], sort_by, ascending)
        
        # Store the result
        result_id = f"sort_result_{len(excel_state['results'])}"
        excel_state["results"][result_id] = sorted_df
        excel_state["active_data"] = sorted_df  # Update active data
        
        return f"Data sorted by {sort_by}. Result ID: {result_id}"
    except Exception as e:
        return f"Error sorting data: {str(e)}"

@tool
def create_pivot(index: List[str], columns: List[str], values: List[str], aggfunc: str = "sum") -> str:
    """
    Create a pivot table from the active data.
    
    Args:
        index: List of columns to use as index
        columns: List of columns to use as pivot columns
        values: List of columns to aggregate
        aggfunc: Aggregation function to use
        
    Returns:
        Summary of pivot table
    """
    if excel_state["active_data"] is None:
        return "No active sheet. Please load a file and select a sheet first."
    
    try:
        pivot_df = create_pivot_table(excel_state["active_data"], index, columns, values, aggfunc)
        
        # Store the result
        result_id = f"pivot_result_{len(excel_state['results'])}"
        excel_state["results"][result_id] = pivot_df
        excel_state["active_data"] = pivot_df  # Update active data
        
        return f"Pivot table created with {len(pivot_df)} rows. Result ID: {result_id}"
    except Exception as e:
        return f"Error creating pivot table: {str(e)}"

@tool
def merge_sheets(sheet_names: List[str], on: Union[str, List[str]], how: str = "inner") -> str:
    """
    Merge multiple sheets based on common columns.
    
    Args:
        sheet_names: List of sheet names to merge
        on: Column(s) to merge on
        how: Type of merge (inner, outer, left, right)
        
    Returns:
        Summary of merged data
    """
    if not excel_state["sheets"]:
        return "No Excel file loaded. Please load a file first."
    
    try:
        merged_df = merge_worksheets(excel_state["sheets"], sheet_names, on, how)
        
        # Store the result
        result_id = f"merge_result_{len(excel_state['results'])}"
        excel_state["results"][result_id] = merged_df
        excel_state["active_data"] = merged_df  # Update active data
        
        return f"Merged {len(sheet_names)} sheets: {len(merged_df)} rows. Result ID: {result_id}"
    except Exception as e:
        return f"Error merging sheets: {str(e)}"

@tool
def filter_by_date(date_column: str, period: str) -> str:
    """
    Filter data by date range based on period specification.
    
    Args:
        date_column: Name of the date column
        period: Period specification (e.g., 'Q3 2024', 'last 6 months')
        
    Returns:
        Summary of filtered data
    """
    if excel_state["active_data"] is None:
        return "No active sheet. Please load a file and select a sheet first."
    
    try:
        # Map column name if needed
        if date_column not in excel_state["active_data"].columns:
            mapped = map_column_names(excel_state["active_data"], [date_column])
            if mapped[date_column] is not None:
                date_column = mapped[date_column]
            else:
                return f"Date column '{date_column}' not found"
        
        # Extract date range
        start_date, end_date = extract_date_range(excel_state["active_data"], date_column, period)
        
        # Ensure date column is datetime type
        df = excel_state["active_data"].copy()
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        
        # Filter by date range
        filtered_df = df[(df[date_column] >= start_date) & (df[date_column] <= end_date)]
        
        # Store the result
        result_id = f"date_filter_result_{len(excel_state['results'])}"
        excel_state["results"][result_id] = filtered_df
        excel_state["active_data"] = filtered_df  # Update active data
        
        return f"Filtered data from {start_date.date()} to {end_date.date()}: {len(filtered_df)} rows. Result ID: {result_id}"
    except Exception as e:
        return f"Error filtering by date: {str(e)}"

@tool
def validate_sheet_data(rules: Dict[str, Dict]) -> str:
    """
    Validate data against rules and return summary of invalid rows.
    
    Args:
        rules: Dictionary of column names and validation rules
        
    Returns:
        Summary of validation results
    """
    if excel_state["active_data"] is None:
        return "No active sheet. Please load a file and select a sheet first."
    
    try:
        invalid_data = validate_data(excel_state["active_data"], rules)
        
        if not invalid_data:
            return "All data is valid according to the specified rules."
        
        summary = {}
        for rule_name, invalid_df in invalid_data.items():
            summary[rule_name] = len(invalid_df)
            
            # Store the invalid data
            result_id = f"invalid_{rule_name}_{len(excel_state['results'])}"
            excel_state["results"][result_id] = invalid_df
        
        return f"Validation results: {json.dumps(summary, indent=2)}"
    except Exception as e:
        return f"Error validating data: {str(e)}"

@tool
def get_data_preview(result_id: Optional[str] = None, rows: int = 5) -> str:
    """
    Get a preview of data from active sheet or a specific result.
    
    Args:
        result_id: ID of the result to preview (if None, use active data)
        rows: Number of rows to preview
        
    Returns:
        Data preview as formatted string
    """
    if result_id is not None:
        if result_id not in excel_state["results"]:
            return f"Result ID '{result_id}' not found. Available results: {list(excel_state['results'].keys())}"
        df = excel_state["results"][result_id]
    elif excel_state["active_data"] is not None:
        df = excel_state["active_data"]
    else:
        return "No active sheet. Please load a file and select a sheet first."
    
    try:
        preview = df.head(rows).to_dict(orient='records')
        return json.dumps(preview, indent=2, default=str)
    except Exception as e:
        return f"Error getting data preview: {str(e)}"

@tool
def export_result(result_id: Optional[str] = None, file_path: Optional[str] = None) -> str:
    """
    Export data to Excel file.
    
    Args:
        result_id: ID of the result to export (if None, use active data)
        file_path: Path to save the Excel file (if None, use temp file)
        
    Returns:
        Path to the exported file
    """
    if result_id is not None:
        if result_id not in excel_state["results"]:
            return f"Result ID '{result_id}' not found. Available results: {list(excel_state['results'].keys())}"
        df = excel_state["results"][result_id]
    elif excel_state["active_data"] is not None:
        df = excel_state["active_data"]
    else:
        return "No active sheet. Please load a file and select a sheet first."
    
    try:
        if file_path is None:
            # Create a temporary file
            fd, file_path = tempfile.mkstemp(suffix='.xlsx')
            os.close(fd)
        
        df.to_excel(file_path, index=False)
        return f"Data exported to {file_path}"
    except Exception as e:
        return f"Error exporting data: {str(e)}"

@tool
def run_complex_query(query: str) -> str:
    """
    Run a complex query against the active data using pandas query syntax.
    
    Args:
        query: Query string in pandas query syntax
        
    Returns:
        Summary of query results
    """
    if excel_state["active_data"] is None:
        return "No active sheet. Please load a file and select a sheet first."
    
    try:
        result_df = excel_state["active_data"].query(query)
        
        # Store the result
        result_id = f"query_result_{len(excel_state['results'])}"
        excel_state["results"][result_id] = result_df
        excel_state["active_data"] = result_df  # Update active data
        
        return f"Query returned {len(result_df)} rows. Result ID: {result_id}"
    except Exception as e:
        return f"Error running query: {str(e)}"

@tool
def get_sheet_statistics(columns: Optional[List[str]] = None) -> str:
    """
    Get descriptive statistics for the active sheet.
    
    Args:
        columns: Optional list of columns to get statistics for
        
    Returns:
        Statistics as JSON string
    """
    if excel_state["active_data"] is None:
        return "No active sheet. Please load a file and select a sheet first."
    
    try:
        df = excel_state["active_data"]
        
        if columns:
            # Map column names if needed
            mapped_columns = []
            for col in columns:
                if col not in df.columns:
                    mapped = map_column_names(df, [col])
                    if mapped[col] is not None:
                        mapped_columns.append(mapped[col])
                else:
                    mapped_columns.append(col)
            
            # Filter to only numeric columns
            numeric_columns = [col for col in mapped_columns if pd.api.types.is_numeric_dtype(df[col])]
            if not numeric_columns:
                return "No numeric columns found in the specified columns."
            
            stats = df[numeric_columns].describe().to_dict()
        else:
            # Filter to only numeric columns
            numeric_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
            if not numeric_columns:
                return "No numeric columns found in the active sheet."
            
            stats = df[numeric_columns].describe().to_dict()
        
        return json.dumps(stats, indent=2, default=str)
    except Exception as e:
        return f"Error getting statistics: {str(e)}"

@tool
def analyze_excel_structure() -> str:
    """
    Analyze the structure of the Excel file and provide insights.
    
    Returns:
        Analysis results as JSON string
    """
    if not excel_state["sheets"]:
        return "No Excel file loaded. Please load a file first."
    
    analysis = {}
    
    for sheet_name, df in excel_state["sheets"].items():
        sheet_analysis = {
            "rows": len(df),
            "columns": len(df.columns),
            "memory_usage": df.memory_usage(deep=True).sum() / (1024 * 1024),  # MB
            "missing_values": df.isna().sum().sum(),
            "missing_percentage": (df.isna().sum().sum() / (len(df) * len(df.columns))) * 100 if len(df) > 0 else 0,
            "column_types": {
                "numeric": sum(1 for col in df.columns if pd.api.types.is_numeric_dtype(df[col])),
                "datetime": sum(1 for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])),
                "categorical": sum(1 for col in df.columns if df[col].nunique() < min(20, len(df) // 10) and not pd.api.types.is_numeric_dtype(df[col])),
                "text": sum(1 for col in df.columns if not pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_datetime64_any_dtype(df[col]) and df[col].nunique() >= min(20, len(df) // 10))
            },
            "potential_primary_keys": [col for col in df.columns if df[col].nunique() == len(df) and df[col].nunique() > 0]
        }
        
        analysis[sheet_name] = sheet_analysis
    
    # Add relationships between sheets if possible
    analysis["potential_relationships"] = []
    sheet_names = list(excel_state["sheets"].keys())
    
    for i in range(len(sheet_names)):
        for j in range(i + 1, len(sheet_names)):
            sheet1 = sheet_names[i]
            sheet2 = sheet_names[j]
            
            # Look for common column names that might indicate relationships
            common_columns = set(excel_state["sheets"][sheet1].columns) & set(excel_state["sheets"][sheet2].columns)
            
            for col in common_columns:
                # Check if the column could be a foreign key
                if col in analysis[sheet1]["potential_primary_keys"] or col in analysis[sheet2]["potential_primary_keys"]:
                    analysis["potential_relationships"].append({
                        "sheet1": sheet1,
                        "sheet2": sheet2,
                        "common_column": col
                    })
    
    return json.dumps(analysis, indent=2, default=str)

@tool
def suggest_operations() -> str:
    """
    Suggest operations based on the structure of the active data.
    
    Returns:
        Suggestions as JSON string
    """
    if excel_state["active_data"] is None:
        return "No active sheet. Please load a file and select a sheet first."
    
    df = excel_state["active_data"]
    suggestions = []
    
    # Check for missing values
    missing_cols = [col for col in df.columns if df[col].isna().sum() > 0]
    if missing_cols:
        suggestions.append({
            "operation": "handle_missing_values",
            "description": f"Handle missing values in columns: {', '.join(missing_cols)}",
            "columns": missing_cols
        })
    
    # Check for date columns that could be used for time series analysis
    date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col]) or 
                 excel_state["column_purposes"].get(excel_state["active_sheet"], {}).get(col) == 'date']
    if date_cols:
        suggestions.append({
            "operation": "time_series_analysis",
            "description": f"Perform time series analysis using date columns: {', '.join(date_cols)}",
            "columns": date_cols
        })
    
    # Check for numeric columns that could be aggregated
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    if len(numeric_cols) >= 2:
        suggestions.append({
            "operation": "aggregate_data",
            "description": f"Aggregate numeric data: {', '.join(numeric_cols[:3])}...",
            "columns": numeric_cols
        })
    
    # Check for categorical columns that could be used for grouping
    categorical_cols = [col for col in df.columns if df[col].nunique() < min(20, len(df) // 10) and not pd.api.types.is_numeric_dtype(df[col])]
    if categorical_cols and numeric_cols:
        suggestions.append({
            "operation": "group_by_analysis",
            "description": f"Group by categorical columns: {', '.join(categorical_cols[:3])}...",
            "columns": categorical_cols
        })
    
    # Check for potential pivot table operations
    if categorical_cols and len(categorical_cols) >= 2 and numeric_cols:
        suggestions.append({
            "operation": "create_pivot",
            "description": f"Create pivot table with index={categorical_cols[0]}, columns={categorical_cols[1]}, values={numeric_cols[0]}",
            "index": categorical_cols[0],
            "columns": categorical_cols[1],
            "values": numeric_cols[0]
        })
    
    return json.dumps(suggestions, indent=2)

# List of all tools for the agent
excel_tools = [
    load_excel_file,
    get_file_status,  # Add the new tool
    switch_sheet,
    get_column_info,
    filter_sheet_data,
    aggregate_sheet_data,
    sort_sheet_data,
    create_pivot,
    merge_sheets,
    filter_by_date,
    validate_sheet_data,
    get_data_preview,
    export_result,
    run_complex_query,
    get_sheet_statistics,
    analyze_excel_structure,
    suggest_operations
] 