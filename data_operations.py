import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
import re
from datetime import datetime, timedelta
from langchain.agents import tool

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@tool
def filter_data(df: pd.DataFrame, conditions: Dict[str, Any]) -> pd.DataFrame:
    """Filter dataframe based on conditions."""
    if not conditions:
        return df
    
    filtered_df = df.copy()
    
    for column, condition in conditions.items():
        if column not in filtered_df.columns:
            continue
        
        try:
            if isinstance(condition, dict):
                for op, value in condition.items():
                    if op in ["eq", "=="]:
                        filtered_df = filtered_df[filtered_df[column] == value]
                    elif op in ["neq", "!="]:
                        filtered_df = filtered_df[filtered_df[column] != value]
                    elif op in ["gt", ">"]:
                        filtered_df = filtered_df[filtered_df[column] > value]
                    elif op in ["gte", ">="]:
                        filtered_df = filtered_df[filtered_df[column] >= value]
                    elif op in ["lt", "<"]:
                        filtered_df = filtered_df[filtered_df[column] < value]
                    elif op in ["lte", "<="]:
                        filtered_df = filtered_df[filtered_df[column] <= value]
                    elif op == "contains":
                        filtered_df = filtered_df[filtered_df[column].astype(str).str.contains(str(value), na=False)]
                    elif op == "in":
                        filtered_df = filtered_df[filtered_df[column].isin(value)]
            else:
                filtered_df = filtered_df[filtered_df[column] == condition]
        except Exception as e:
            logger.error(f"Error filtering on column '{column}': {str(e)}")
    
    return filtered_df

@tool
def filter_by_date_range(df: pd.DataFrame, date_column: str, 
                       start_date: Optional[str] = None, 
                       end_date: Optional[str] = None,
                       period: Optional[str] = None) -> pd.DataFrame:
    """Filter dataframe by date range."""
    if date_column not in df.columns:
        return df
    
    # Ensure date column is datetime type
    try:
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    except:
        return df
    
    # Handle predefined periods
    if period:
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        if period == "today":
            start_date, end_date = today, today + timedelta(days=1)
        elif period == "yesterday":
            start_date, end_date = today - timedelta(days=1), today
        elif period == "last_7_days":
            start_date, end_date = today - timedelta(days=7), today
        elif period == "last_30_days":
            start_date, end_date = today - timedelta(days=30), today
        elif period == "this_month":
            start_date = today.replace(day=1)
            end_date = (today.replace(day=28) + timedelta(days=4)).replace(day=1)
        elif period == "last_month":
            end_date = today.replace(day=1)
            start_date = (end_date - timedelta(days=1)).replace(day=1)
        elif re.match(r"q[1-4]_\d{4}", period):
            quarter = int(period[1])
            year = int(period[3:])
            start_month = (quarter - 1) * 3 + 1
            end_month = quarter * 3 + 1
            start_date = datetime(year, start_month, 1)
            end_date = datetime(year, end_month, 1)
    
    # Apply date filters
    filtered_df = df.copy()
    
    if start_date:
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        filtered_df = filtered_df[filtered_df[date_column] >= start_date]
    
    if end_date:
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        filtered_df = filtered_df[filtered_df[date_column] < end_date]
    
    return filtered_df

@tool
def aggregate_data(df: pd.DataFrame, group_by: List[str], 
                  aggregations: Dict[str, str]) -> pd.DataFrame:
    """Aggregate data by grouping and applying aggregation functions."""
    # Validate group_by columns
    valid_group_by = [col for col in group_by if col in df.columns]
    if not valid_group_by:
        return df
    
    # Validate aggregation columns
    valid_aggs = {col: agg for col, agg in aggregations.items() if col in df.columns}
    if not valid_aggs:
        return df
    
    try:
        # Perform groupby and aggregation
        result = df.groupby(valid_group_by).agg(valid_aggs)
        
        # Flatten multi-level column names if needed
        if isinstance(result.columns, pd.MultiIndex):
            result.columns = [f"{col[0]}_{col[1]}" for col in result.columns]
        
        # Reset index
        return result.reset_index()
    except:
        return df

@tool
def sort_data(df: pd.DataFrame, sort_by: List[Any]) -> pd.DataFrame:
    """Sort dataframe by specified columns."""
    if not sort_by:
        return df
    
    columns = []
    ascending = []
    
    for item in sort_by:
        if isinstance(item, dict):
            for col, direction in item.items():
                if col in df.columns:
                    columns.append(col)
                    ascending.append(direction.lower() != "desc")
        elif item in df.columns:
            columns.append(item)
            ascending.append(True)
    
    if not columns:
        return df
    
    try:
        return df.sort_values(by=columns, ascending=ascending)
    except:
        return df

@tool
def pivot_table(df: pd.DataFrame, index: List[str], columns: List[str], 
               values: List[str], aggfunc: str = "mean") -> pd.DataFrame:
    """Create a pivot table from the dataframe."""
    # Validate columns
    valid_index = [col for col in index if col in df.columns]
    valid_columns = [col for col in columns if col in df.columns]
    valid_values = [col for col in values if col in df.columns]
    
    if not valid_index or not valid_values:
        return df
    
    try:
        # Create pivot table
        pivot = pd.pivot_table(
            df,
            index=valid_index,
            columns=valid_columns if valid_columns else None,
            values=valid_values,
            aggfunc=aggfunc
        )
        
        # Reset index for easier handling
        pivot = pivot.reset_index()
        
        # Flatten column hierarchy if needed
        if isinstance(pivot.columns, pd.MultiIndex):
            pivot.columns = ['_'.join(str(i) for i in col if i) for col in pivot.columns]
        
        return pivot
    except:
        return df

@tool
def merge_worksheets(df1: pd.DataFrame, df2: pd.DataFrame, 
                    how: str = "inner", on: Optional[List[str]] = None,
                    left_on: Optional[List[str]] = None, 
                    right_on: Optional[List[str]] = None) -> pd.DataFrame:
    """Merge two dataframes."""
    try:
        return pd.merge(df1, df2, how=how, on=on, left_on=left_on, right_on=right_on)
    except Exception as e:
        logger.error(f"Error merging dataframes: {str(e)}")
        return df1

@tool
def apply_formula(df: pd.DataFrame, formula: str, output_column: str) -> pd.DataFrame:
    """Apply a formula to create a new column."""
    try:
        result_df = df.copy()
        
        # Replace column names with df['column_name'] for evaluation
        eval_formula = formula
        for col in df.columns:
            pattern = r'\b' + re.escape(col) + r'\b'
            eval_formula = re.sub(pattern, f"df['{col}']", eval_formula)
        
        # Evaluate the formula
        result_df[output_column] = eval(eval_formula)
        return result_df
    except:
        return df 