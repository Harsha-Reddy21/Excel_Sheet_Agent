import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union, Tuple
import os
import re
from fuzzywuzzy import process

def read_excel_file(file_path: str) -> Dict[str, pd.DataFrame]:
    """
    Read an Excel file and return a dictionary of DataFrames, one for each sheet.
    Handles large files by using optimized pandas options.
    
    Args:
        file_path: Path to the Excel file
        
    Returns:
        Dictionary with sheet names as keys and pandas DataFrames as values
    """
    try:
        # Create ExcelFile object without specifying engine - pandas will choose appropriate one
        excel_file = pd.ExcelFile(file_path)
        sheets = {}
        
        for sheet_name in excel_file.sheet_names:
            # Read sheet without specifying engine again - use the one from ExcelFile
            df = pd.read_excel(
                excel_file, 
                sheet_name=sheet_name
                # Don't specify engine here - it's already set in ExcelFile
            )
            sheets[sheet_name] = df
            
        return sheets
    except Exception as e:
        raise Exception(f"Error reading Excel file: {str(e)}")

def get_sheet_info(sheets: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
    """
    Get information about each sheet in the Excel file
    
    Args:
        sheets: Dictionary of sheet names and DataFrames
        
    Returns:
        List of dictionaries with sheet information
    """
    sheet_info = []
    for name, df in sheets.items():
        info = {
            "name": name,
            "rows": len(df),
            "columns": list(df.columns),
            "data_types": {col: str(df[col].dtype) for col in df.columns}
        }
        sheet_info.append(info)
    return sheet_info

def map_column_names(df: pd.DataFrame, query_columns: List[str]) -> Dict[str, str]:
    """
    Map query column names to actual column names in the DataFrame using fuzzy matching
    
    Args:
        df: DataFrame containing the actual columns
        query_columns: List of column names from the user query
        
    Returns:
        Dictionary mapping query column names to actual column names
    """
    actual_columns = list(df.columns)
    column_mapping = {}
    
    # Dictionary of common business term synonyms - expanded to be more generic
    synonyms = {
        "quantity": ["qty", "count", "amount", "number", "volume", "units"],
        "revenue": ["sales", "income", "earnings", "rev", "turnover", "proceeds"],
        "customer": ["client", "buyer", "purchaser", "consumer", "user", "account"],
        "product": ["item", "merchandise", "goods", "sku", "article", "commodity"],
        "date": ["time", "period", "day", "timestamp", "datetime", "when"],
        "region": ["area", "location", "territory", "zone", "district", "geo"],
        "price": ["cost", "rate", "value", "amount", "fee", "charge"],
        "name": ["title", "label", "designation", "term", "handle"],
        "address": ["location", "place", "residence", "domicile"],
        "email": ["mail", "e-mail", "electronic mail", "contact"],
        "phone": ["telephone", "mobile", "cell", "contact", "number"],
        "total": ["sum", "aggregate", "gross", "entirety", "complete"],
        "average": ["mean", "avg", "typical", "median", "norm"],
        "maximum": ["max", "highest", "peak", "top", "ceiling"],
        "minimum": ["min", "lowest", "bottom", "floor", "least"],
        "percentage": ["percent", "pct", "%", "proportion", "ratio"],
        "status": ["state", "condition", "standing", "position"],
        "category": ["type", "class", "group", "classification", "segment"],
        "description": ["desc", "details", "info", "specification", "explanation"]
    }
    
    # Normalize column names for better matching
    def normalize_name(name):
        # Convert to lowercase and remove special characters
        name = re.sub(r'[^a-zA-Z0-9]', ' ', str(name).lower())
        # Remove extra spaces
        name = re.sub(r'\s+', ' ', name).strip()
        return name
    
    normalized_actual = {normalize_name(col): col for col in actual_columns}
    
    for query_col in query_columns:
        normalized_query = normalize_name(query_col)
        
        # Direct match
        if normalized_query in normalized_actual:
            column_mapping[query_col] = normalized_actual[normalized_query]
            continue
            
        # Check synonyms
        matched = False
        for key, values in synonyms.items():
            if normalized_query == key or normalized_query in values:
                for actual_norm, actual_col in normalized_actual.items():
                    if actual_norm == key or any(syn in actual_norm for syn in values):
                        column_mapping[query_col] = actual_col
                        matched = True
                        break
            if matched:
                break
                
        # Fuzzy matching as a fallback - more aggressive matching
        if not matched:
            match, score = process.extractOne(normalized_query, list(normalized_actual.keys()))
            if score > 60:  # Lower threshold for more flexible matching
                column_mapping[query_col] = normalized_actual[match]
            else:
                # If no good match, just use the original column name
                # This allows the query to proceed and fail naturally if needed
                column_mapping[query_col] = query_col
    
    return column_mapping

def chunk_dataframe(df: pd.DataFrame, chunk_size: int = 1000) -> List[pd.DataFrame]:
    """
    Split a large DataFrame into manageable chunks
    
    Args:
        df: DataFrame to chunk
        chunk_size: Number of rows per chunk
        
    Returns:
        List of DataFrame chunks
    """
    return [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]

def filter_data(df: pd.DataFrame, conditions: Dict[str, Any]) -> pd.DataFrame:
    """
    Filter DataFrame based on conditions
    
    Args:
        df: DataFrame to filter
        conditions: Dictionary of column-condition pairs
        
    Returns:
        Filtered DataFrame
    """
    filtered_df = df.copy()
    
    for column, condition in conditions.items():
        if column not in df.columns:
            # Try to map column name
            mapped_columns = map_column_names(df, [column])
            if mapped_columns[column] is not None:
                column = mapped_columns[column]
            else:
                continue  # Skip this condition if column not found
        
        # Handle different condition types
        if isinstance(condition, dict):
            op = condition.get('operator')
            value = condition.get('value')
            
            if op == '>' or op == 'greater than':
                filtered_df = filtered_df[filtered_df[column] > value]
            elif op == '<' or op == 'less than':
                filtered_df = filtered_df[filtered_df[column] < value]
            elif op == '>=' or op == 'greater than or equal to':
                filtered_df = filtered_df[filtered_df[column] >= value]
            elif op == '<=' or op == 'less than or equal to':
                filtered_df = filtered_df[filtered_df[column] <= value]
            elif op == '==' or op == 'equal to':
                filtered_df = filtered_df[filtered_df[column] == value]
            elif op == '!=' or op == 'not equal to':
                filtered_df = filtered_df[filtered_df[column] != value]
            elif op == 'contains':
                filtered_df = filtered_df[filtered_df[column].astype(str).str.contains(str(value), na=False)]
            elif op == 'between':
                if isinstance(value, list) and len(value) == 2:
                    filtered_df = filtered_df[(filtered_df[column] >= value[0]) & (filtered_df[column] <= value[1])]
        elif isinstance(condition, list):
            filtered_df = filtered_df[filtered_df[column].isin(condition)]
        else:
            # Simple equality check
            filtered_df = filtered_df[filtered_df[column] == condition]
    
    return filtered_df

def aggregate_data(df: pd.DataFrame, group_by: List[str], aggregations: Dict[str, str]) -> pd.DataFrame:
    """
    Aggregate DataFrame by grouping and applying aggregation functions
    
    Args:
        df: DataFrame to aggregate
        group_by: List of columns to group by
        aggregations: Dictionary mapping columns to aggregation functions
        
    Returns:
        Aggregated DataFrame
    """
    # Map column names if needed
    mapped_group_by = []
    for col in group_by:
        if col not in df.columns:
            mapped = map_column_names(df, [col])
            if mapped[col] is not None:
                mapped_group_by.append(mapped[col])
        else:
            mapped_group_by.append(col)
    
    mapped_aggs = {}
    for col, agg in aggregations.items():
        if col not in df.columns:
            mapped = map_column_names(df, [col])
            if mapped[col] is not None:
                mapped_aggs[mapped[col]] = agg
        else:
            mapped_aggs[col] = agg
    
    # Skip if no valid columns found
    if not mapped_group_by or not mapped_aggs:
        return df
    
    return df.groupby(mapped_group_by).agg(mapped_aggs).reset_index()

def sort_data(df: pd.DataFrame, sort_by: List[str], ascending: Union[bool, List[bool]] = True) -> pd.DataFrame:
    """
    Sort DataFrame by specified columns
    
    Args:
        df: DataFrame to sort
        sort_by: List of columns to sort by
        ascending: Whether to sort in ascending order
        
    Returns:
        Sorted DataFrame
    """
    # Map column names if needed
    mapped_sort_by = []
    for col in sort_by:
        if col not in df.columns:
            mapped = map_column_names(df, [col])
            if mapped[col] is not None:
                mapped_sort_by.append(mapped[col])
        else:
            mapped_sort_by.append(col)
    
    if not mapped_sort_by:
        return df
    
    return df.sort_values(by=mapped_sort_by, ascending=ascending)

def create_pivot_table(df: pd.DataFrame, index: List[str], columns: List[str], 
                      values: List[str], aggfunc: str = 'sum') -> pd.DataFrame:
    """
    Create a pivot table from DataFrame
    
    Args:
        df: Source DataFrame
        index: List of columns to use as index
        columns: List of columns to use as pivot columns
        values: List of columns to aggregate
        aggfunc: Aggregation function to use
        
    Returns:
        Pivot table as DataFrame
    """
    # Map column names
    mapped_index = []
    for col in index:
        if col not in df.columns:
            mapped = map_column_names(df, [col])
            if mapped[col] is not None:
                mapped_index.append(mapped[col])
        else:
            mapped_index.append(col)
    
    mapped_columns = []
    for col in columns:
        if col not in df.columns:
            mapped = map_column_names(df, [col])
            if mapped[col] is not None:
                mapped_columns.append(mapped[col])
        else:
            mapped_columns.append(col)
    
    mapped_values = []
    for col in values:
        if col not in df.columns:
            mapped = map_column_names(df, [col])
            if mapped[col] is not None:
                mapped_values.append(mapped[col])
        else:
            mapped_values.append(col)
    
    # Skip if no valid columns found
    if not mapped_index or not mapped_values:
        return df
    
    # Convert aggfunc string to actual function
    if aggfunc == 'sum':
        agg_function = np.sum
    elif aggfunc == 'mean' or aggfunc == 'average':
        agg_function = np.mean
    elif aggfunc == 'count':
        agg_function = len
    elif aggfunc == 'min':
        agg_function = np.min
    elif aggfunc == 'max':
        agg_function = np.max
    else:
        agg_function = np.sum  # Default
    
    pivot = pd.pivot_table(
        df, 
        index=mapped_index, 
        columns=mapped_columns if mapped_columns else None, 
        values=mapped_values,
        aggfunc=agg_function
    )
    
    return pivot.reset_index()

def merge_worksheets(sheets: Dict[str, pd.DataFrame], sheet_names: List[str], 
                    on: Union[str, List[str]], how: str = 'inner') -> pd.DataFrame:
    """
    Merge multiple worksheets based on common columns
    
    Args:
        sheets: Dictionary of sheet names and DataFrames
        sheet_names: List of sheet names to merge
        on: Column(s) to merge on
        how: Type of merge (inner, outer, left, right)
        
    Returns:
        Merged DataFrame
    """
    if len(sheet_names) < 2:
        raise ValueError("Need at least two sheets to merge")
    
    if sheet_names[0] not in sheets:
        raise ValueError(f"Sheet {sheet_names[0]} not found")
    
    result = sheets[sheet_names[0]]
    
    for name in sheet_names[1:]:
        if name not in sheets:
            raise ValueError(f"Sheet {name} not found")
        
        # Try to map column names if 'on' columns don't exist
        if isinstance(on, str):
            on_cols = [on]
        else:
            on_cols = on
            
        left_cols = []
        right_cols = []
        
        for col in on_cols:
            if col not in result.columns:
                left_mapped = map_column_names(result, [col])
                left_col = left_mapped[col] if left_mapped[col] is not None else col
                left_cols.append(left_col)
            else:
                left_cols.append(col)
                
            if col not in sheets[name].columns:
                right_mapped = map_column_names(sheets[name], [col])
                right_col = right_mapped[col] if right_mapped[col] is not None else col
                right_cols.append(right_col)
            else:
                right_cols.append(col)
        
        # Perform the merge
        if len(left_cols) == 1 and len(right_cols) == 1:
            result = pd.merge(result, sheets[name], left_on=left_cols[0], 
                             right_on=right_cols[0], how=how, suffixes=('', f'_{name}'))
        else:
            result = pd.merge(result, sheets[name], left_on=left_cols, 
                             right_on=right_cols, how=how, suffixes=('', f'_{name}'))
    
    return result

def extract_date_range(df: pd.DataFrame, date_col: str, period: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Extract date range based on period specification (e.g., 'Q3 2024')
    
    Args:
        df: DataFrame with date column
        date_col: Name of the date column
        period: Period specification (e.g., 'Q3 2024', 'last 6 months')
        
    Returns:
        Tuple of start and end dates
    """
    # Map column name if needed
    if date_col not in df.columns:
        mapped = map_column_names(df, [date_col])
        if mapped[date_col] is not None:
            date_col = mapped[date_col]
        else:
            raise ValueError(f"Date column '{date_col}' not found")
    
    # Ensure date column is datetime type
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except:
            raise ValueError(f"Could not convert column '{date_col}' to datetime")
    
    # Current date for relative calculations
    current_date = pd.Timestamp.now()
    
    # Parse period string
    if re.match(r'Q[1-4]\s+\d{4}', period):
        # Quarter specification (e.g., "Q3 2024")
        quarter = int(period[1])
        year = int(period[3:])
        start_date = pd.Timestamp(f"{year}-{(quarter-1)*3+1}-01")
        end_date = start_date + pd.DateOffset(months=3) - pd.DateOffset(days=1)
    
    elif re.match(r'last\s+\d+\s+months?', period, re.IGNORECASE):
        # Last N months
        months = int(re.search(r'last\s+(\d+)', period).group(1))
        end_date = current_date
        start_date = end_date - pd.DateOffset(months=months)
    
    elif re.match(r'last\s+\d+\s+days?', period, re.IGNORECASE):
        # Last N days
        days = int(re.search(r'last\s+(\d+)', period).group(1))
        end_date = current_date
        start_date = end_date - pd.DateOffset(days=days)
    
    elif re.match(r'last\s+year', period, re.IGNORECASE):
        # Last year
        end_date = current_date
        start_date = end_date - pd.DateOffset(years=1)
    
    elif re.match(r'year\s+\d{4}', period, re.IGNORECASE):
        # Specific year
        year = int(re.search(r'year\s+(\d{4})', period).group(1))
        start_date = pd.Timestamp(f"{year}-01-01")
        end_date = pd.Timestamp(f"{year}-12-31")
    
    else:
        # Default to all dates
        start_date = df[date_col].min()
        end_date = df[date_col].max()
    
    return start_date, end_date

def validate_data(df: pd.DataFrame, rules: Dict[str, Dict]) -> Dict[str, pd.DataFrame]:
    """
    Validate data against rules and return invalid rows
    
    Args:
        df: DataFrame to validate
        rules: Dictionary of column names and validation rules
        
    Returns:
        Dictionary of rule names and DataFrames with invalid rows
    """
    invalid_data = {}
    
    for column, rule_dict in rules.items():
        # Map column name if needed
        if column not in df.columns:
            mapped = map_column_names(df, [column])
            if mapped[column] is not None:
                column = mapped[column]
            else:
                continue  # Skip if column not found
        
        rule_type = rule_dict.get('type')
        
        if rule_type == 'not_null':
            invalid = df[df[column].isna()]
            if not invalid.empty:
                invalid_data[f"{column}_not_null"] = invalid
        
        elif rule_type == 'unique':
            duplicates = df[df.duplicated(subset=[column], keep=False)]
            if not duplicates.empty:
                invalid_data[f"{column}_unique"] = duplicates
        
        elif rule_type == 'range':
            min_val = rule_dict.get('min')
            max_val = rule_dict.get('max')
            
            if min_val is not None and max_val is not None:
                invalid = df[(df[column] < min_val) | (df[column] > max_val)]
            elif min_val is not None:
                invalid = df[df[column] < min_val]
            elif max_val is not None:
                invalid = df[df[column] > max_val]
            else:
                continue
                
            if not invalid.empty:
                invalid_data[f"{column}_range"] = invalid
        
        elif rule_type == 'pattern':
            pattern = rule_dict.get('pattern')
            if pattern:
                invalid = df[~df[column].astype(str).str.match(pattern)]
                if not invalid.empty:
                    invalid_data[f"{column}_pattern"] = invalid
    
    return invalid_data

def detect_data_types(df: pd.DataFrame) -> Dict[str, str]:
    """
    Detect and return data types for each column
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary of column names and detected data types
    """
    type_mapping = {}
    
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            if pd.api.types.is_integer_dtype(df[column]):
                type_mapping[column] = 'integer'
            else:
                type_mapping[column] = 'float'
        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            type_mapping[column] = 'datetime'
        elif pd.api.types.is_bool_dtype(df[column]):
            type_mapping[column] = 'boolean'
        else:
            # Check if it could be a date
            try:
                pd.to_datetime(df[column], errors='raise')
                type_mapping[column] = 'potential_datetime'
            except:
                # Check if it's categorical (few unique values)
                if df[column].nunique() < min(20, len(df) // 10):  # Dynamic threshold
                    type_mapping[column] = 'categorical'
                else:
                    type_mapping[column] = 'text'
    
    return type_mapping

def infer_column_purpose(df: pd.DataFrame) -> Dict[str, str]:
    """
    Infer the likely purpose of each column based on name and content
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary of column names and inferred purposes
    """
    purposes = {}
    
    # Common patterns to look for in column names
    patterns = {
        'id': [r'id$', r'^id', r'code', r'key', r'number', r'#', r'num'],
        'date': [r'date', r'time', r'day', r'month', r'year', r'dt'],
        'name': [r'name', r'title', r'label'],
        'quantity': [r'qty', r'quantity', r'count', r'number', r'amount'],
        'price': [r'price', r'cost', r'rate', r'fee', r'charge'],
        'total': [r'total', r'sum', r'amount'],
        'category': [r'category', r'type', r'group', r'class'],
        'status': [r'status', r'state', r'condition'],
        'email': [r'email', r'mail', r'e-mail'],
        'phone': [r'phone', r'tel', r'mobile', r'cell'],
        'address': [r'address', r'location', r'place']
    }
    
    for column in df.columns:
        col_lower = str(column).lower()
        
        # Check patterns
        purpose_found = False
        for purpose, pattern_list in patterns.items():
            if any(re.search(pattern, col_lower) for pattern in pattern_list):
                purposes[column] = purpose
                purpose_found = True
                break
        
        if not purpose_found:
            # Fallback based on data type
            dtype = df[column].dtype
            if pd.api.types.is_numeric_dtype(dtype):
                purposes[column] = 'numeric'
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                purposes[column] = 'date'
            else:
                purposes[column] = 'text'
    
    return purposes 