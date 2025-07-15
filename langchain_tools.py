from langchain.tools import BaseTool, tool
from typing import Dict, List, Any, Optional, Union, Tuple, Type
import pandas as pd
import logging
from pydantic import BaseModel, Field

# Import local modules
from excel_reader import read_excel_file, chunk_dataframe, convert_date_columns
from data_operations import (
    filter_data, filter_by_date_range, aggregate_data, 
    sort_data, pivot_table, merge_worksheets, apply_formula
)
from column_mapper import map_column_names, llm_column_mapping
from visualization import (
    create_bar_chart, create_line_chart, create_pie_chart, 
    create_scatter_plot, create_histogram, create_heatmap
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define input/output schemas for tools
class ReadWorksheetInput(BaseModel):
    file_path: str = Field(..., description="Path to the Excel file")
    sheet_name: Optional[str] = Field(None, description="Name of the worksheet to read (None for all sheets)")

class FilterDataInput(BaseModel):
    sheet_name: str = Field(..., description="Name of the worksheet to filter")
    conditions: Dict[str, Any] = Field(..., description="Filter conditions as column:value pairs")

class DateFilterInput(BaseModel):
    sheet_name: str = Field(..., description="Name of the worksheet to filter")
    date_column: str = Field(..., description="Column containing dates")
    start_date: Optional[str] = Field(None, description="Start date (inclusive)")
    end_date: Optional[str] = Field(None, description="End date (inclusive)")
    period: Optional[str] = Field(None, description="Predefined period (e.g., 'last_7_days')")

class AggregateDataInput(BaseModel):
    sheet_name: str = Field(..., description="Name of the worksheet to aggregate")
    group_by: List[str] = Field(..., description="Columns to group by")
    aggregations: Dict[str, str] = Field(..., description="Aggregation functions to apply")

class SortDataInput(BaseModel):
    sheet_name: str = Field(..., description="Name of the worksheet to sort")
    sort_by: List[Union[str, Dict[str, str]]] = Field(..., description="Columns to sort by")

class PivotTableInput(BaseModel):
    sheet_name: str = Field(..., description="Name of the worksheet to pivot")
    index: List[str] = Field(..., description="Columns to use as index")
    columns: List[str] = Field(..., description="Columns to use as columns")
    values: List[str] = Field(..., description="Columns to aggregate")
    aggfunc: str = Field("mean", description="Aggregation function to apply")

class MergeWorksheetsInput(BaseModel):
    left_sheet: str = Field(..., description="Name of the left worksheet")
    right_sheet: str = Field(..., description="Name of the right worksheet")
    how: str = Field("inner", description="Type of merge")
    on: Optional[List[str]] = Field(None, description="Columns to merge on")
    left_on: Optional[List[str]] = Field(None, description="Columns from left sheet")
    right_on: Optional[List[str]] = Field(None, description="Columns from right sheet")

class FormulaInput(BaseModel):
    sheet_name: str = Field(..., description="Name of the worksheet to apply formula to")
    formula: str = Field(..., description="Formula to apply")
    output_column: str = Field(..., description="Name for the output column")

class ChartInput(BaseModel):
    sheet_name: str = Field(..., description="Name of the worksheet to create chart from")
    chart_type: str = Field(..., description="Type of chart to create")
    x_column: Optional[str] = Field(None, description="Column for x-axis")
    y_columns: Optional[List[str]] = Field(None, description="Columns for y-axis")
    title: Optional[str] = Field(None, description="Chart title")

# Global variable to store dataframes
dataframes = {}

@tool
def read_worksheet(file_path: str, sheet_name: Optional[str] = None) -> str:
    """
    Read data from an Excel worksheet.
    
    Args:
        file_path: Path to the Excel file
        sheet_name: Name of the worksheet to read (None for all sheets)
        
    Returns:
        Summary of the data read
    """
    global dataframes
    
    try:
        # Read Excel file
        result = read_excel_file(file_path)
        
        # Store dataframes globally
        dataframes = result
        
        # Convert date columns
        for sheet, df in dataframes.items():
            dataframes[sheet] = convert_date_columns(df)
        
        # Prepare summary
        summary = []
        for sheet, df in dataframes.items():
            if sheet_name is None or sheet == sheet_name:
                summary.append(f"Sheet: {sheet}")
                summary.append(f"Rows: {len(df)}")
                summary.append(f"Columns: {', '.join(df.columns)}")
                summary.append("")
        
        return "\n".join(summary)
    
    except Exception as e:
        logger.error(f"Error reading worksheet: {str(e)}")
        return f"Error: {str(e)}"

@tool
def filter_data_tool(sheet_name: str, conditions: Dict[str, Any]) -> str:
    """
    Filter data based on conditions.
    
    Args:
        sheet_name: Name of the worksheet to filter
        conditions: Filter conditions as column:value pairs
        
    Returns:
        Summary of the filtered data
    """
    global dataframes
    
    try:
        if sheet_name not in dataframes:
            return f"Error: Sheet '{sheet_name}' not found"
        
        df = dataframes[sheet_name]
        
        # Apply filter
        filtered_df = filter_data(df, conditions)
        
        # Store result
        dataframes[f"{sheet_name}_filtered"] = filtered_df
        
        return f"Filtered {sheet_name} from {len(df)} to {len(filtered_df)} rows. Result stored as '{sheet_name}_filtered'"
    
    except Exception as e:
        logger.error(f"Error filtering data: {str(e)}")
        return f"Error: {str(e)}"

@tool
def filter_by_date_tool(sheet_name: str, date_column: str, 
                        start_date: Optional[str] = None, 
                        end_date: Optional[str] = None,
                        period: Optional[str] = None) -> str:
    """
    Filter data by date range.
    
    Args:
        sheet_name: Name of the worksheet to filter
        date_column: Column containing dates
        start_date: Start date (inclusive)
        end_date: End date (inclusive)
        period: Predefined period (e.g., 'last_7_days')
        
    Returns:
        Summary of the filtered data
    """
    global dataframes
    
    try:
        if sheet_name not in dataframes:
            return f"Error: Sheet '{sheet_name}' not found"
        
        df = dataframes[sheet_name]
        
        # Apply date filter
        filtered_df = filter_by_date_range(df, date_column, start_date, end_date, period)
        
        # Store result
        dataframes[f"{sheet_name}_date_filtered"] = filtered_df
        
        return f"Date filtered {sheet_name} from {len(df)} to {len(filtered_df)} rows. Result stored as '{sheet_name}_date_filtered'"
    
    except Exception as e:
        logger.error(f"Error filtering by date: {str(e)}")
        return f"Error: {str(e)}"

@tool
def aggregate_data_tool(sheet_name: str, group_by: List[str], 
                       aggregations: Dict[str, str]) -> str:
    """
    Aggregate data by grouping and applying aggregation functions.
    
    Args:
        sheet_name: Name of the worksheet to aggregate
        group_by: Columns to group by
        aggregations: Aggregation functions to apply
        
    Returns:
        Summary of the aggregated data
    """
    global dataframes
    
    try:
        if sheet_name not in dataframes:
            return f"Error: Sheet '{sheet_name}' not found"
        
        df = dataframes[sheet_name]
        
        # Apply aggregation
        agg_df = aggregate_data(df, group_by, aggregations)
        
        # Store result
        dataframes[f"{sheet_name}_aggregated"] = agg_df
        
        return f"Aggregated {sheet_name} from {len(df)} to {len(agg_df)} rows. Result stored as '{sheet_name}_aggregated'"
    
    except Exception as e:
        logger.error(f"Error aggregating data: {str(e)}")
        return f"Error: {str(e)}"

@tool
def sort_data_tool(sheet_name: str, sort_by: List[Union[str, Dict[str, str]]]) -> str:
    """
    Sort data by specified columns.
    
    Args:
        sheet_name: Name of the worksheet to sort
        sort_by: Columns to sort by
        
    Returns:
        Summary of the sorted data
    """
    global dataframes
    
    try:
        if sheet_name not in dataframes:
            return f"Error: Sheet '{sheet_name}' not found"
        
        df = dataframes[sheet_name]
        
        # Apply sorting
        sorted_df = sort_data(df, sort_by)
        
        # Store result
        dataframes[f"{sheet_name}_sorted"] = sorted_df
        
        return f"Sorted {sheet_name}. Result stored as '{sheet_name}_sorted'"
    
    except Exception as e:
        logger.error(f"Error sorting data: {str(e)}")
        return f"Error: {str(e)}"

@tool
def pivot_table_tool(sheet_name: str, index: List[str], columns: List[str], 
                    values: List[str], aggfunc: str = "mean") -> str:
    """
    Create a pivot table.
    
    Args:
        sheet_name: Name of the worksheet to pivot
        index: Columns to use as index
        columns: Columns to use as columns
        values: Columns to aggregate
        aggfunc: Aggregation function to apply
        
    Returns:
        Summary of the pivot table
    """
    global dataframes
    
    try:
        if sheet_name not in dataframes:
            return f"Error: Sheet '{sheet_name}' not found"
        
        df = dataframes[sheet_name]
        
        # Create pivot table
        pivot_df = pivot_table(df, index, columns, values, aggfunc)
        
        # Store result
        dataframes[f"{sheet_name}_pivot"] = pivot_df
        
        return f"Created pivot table from {sheet_name} with shape {pivot_df.shape}. Result stored as '{sheet_name}_pivot'"
    
    except Exception as e:
        logger.error(f"Error creating pivot table: {str(e)}")
        return f"Error: {str(e)}"

@tool
def merge_worksheets_tool(left_sheet: str, right_sheet: str, how: str = "inner",
                         on: Optional[List[str]] = None, 
                         left_on: Optional[List[str]] = None,
                         right_on: Optional[List[str]] = None) -> str:
    """
    Merge two worksheets.
    
    Args:
        left_sheet: Name of the left worksheet
        right_sheet: Name of the right worksheet
        how: Type of merge
        on: Columns to merge on
        left_on: Columns from left sheet
        right_on: Columns from right sheet
        
    Returns:
        Summary of the merged data
    """
    global dataframes
    
    try:
        if left_sheet not in dataframes:
            return f"Error: Sheet '{left_sheet}' not found"
        
        if right_sheet not in dataframes:
            return f"Error: Sheet '{right_sheet}' not found"
        
        left_df = dataframes[left_sheet]
        right_df = dataframes[right_sheet]
        
        # Merge dataframes
        merged_df = merge_worksheets(left_df, right_df, how, on, left_on, right_on)
        
        # Store result
        result_name = f"{left_sheet}_{right_sheet}_merged"
        dataframes[result_name] = merged_df
        
        return f"Merged {left_sheet} and {right_sheet} into {len(merged_df)} rows. Result stored as '{result_name}'"
    
    except Exception as e:
        logger.error(f"Error merging worksheets: {str(e)}")
        return f"Error: {str(e)}"

@tool
def apply_formula_tool(sheet_name: str, formula: str, output_column: str) -> str:
    """
    Apply a formula to create a new column.
    
    Args:
        sheet_name: Name of the worksheet to apply formula to
        formula: Formula to apply
        output_column: Name for the output column
        
    Returns:
        Summary of the operation
    """
    global dataframes
    
    try:
        if sheet_name not in dataframes:
            return f"Error: Sheet '{sheet_name}' not found"
        
        df = dataframes[sheet_name]
        
        # Apply formula
        result_df = apply_formula(df, formula, output_column)
        
        # Store result
        dataframes[sheet_name] = result_df
        
        return f"Applied formula to create column '{output_column}' in {sheet_name}"
    
    except Exception as e:
        logger.error(f"Error applying formula: {str(e)}")
        return f"Error: {str(e)}"

@tool
def create_chart_tool(sheet_name: str, chart_type: str, x_column: Optional[str] = None,
                     y_columns: Optional[List[str]] = None, title: Optional[str] = None) -> str:
    """
    Create a chart from data.
    
    Args:
        sheet_name: Name of the worksheet to create chart from
        chart_type: Type of chart to create
        x_column: Column for x-axis
        y_columns: Columns for y-axis
        title: Chart title
        
    Returns:
        Base64 encoded image or error message
    """
    global dataframes
    
    try:
        if sheet_name not in dataframes:
            return f"Error: Sheet '{sheet_name}' not found"
        
        df = dataframes[sheet_name]
        
        # Create chart based on type
        if chart_type == "bar" and x_column and y_columns and len(y_columns) > 0:
            return create_bar_chart(df, x_column, y_columns[0], title)
        
        elif chart_type == "line" and x_column and y_columns:
            return create_line_chart(df, x_column, y_columns, title)
        
        elif chart_type == "pie" and x_column and y_columns and len(y_columns) > 0:
            return create_pie_chart(df, x_column, y_columns[0], title)
        
        elif chart_type == "scatter" and x_column and y_columns and len(y_columns) > 0:
            color_column = y_columns[1] if len(y_columns) > 1 else None
            return create_scatter_plot(df, x_column, y_columns[0], color_column, title)
        
        elif chart_type == "histogram" and y_columns and len(y_columns) > 0:
            return create_histogram(df, y_columns[0], title=title)
        
        elif chart_type == "heatmap":
            return create_heatmap(df, y_columns, title)
        
        else:
            return f"Error: Invalid chart parameters for {chart_type}"
    
    except Exception as e:
        logger.error(f"Error creating chart: {str(e)}")
        return f"Error: {str(e)}"

@tool
def write_results(sheet_name: str, output_file: str) -> str:
    """
    Write results to an Excel file.
    
    Args:
        sheet_name: Name of the worksheet to write
        output_file: Path to the output file
        
    Returns:
        Summary of the operation
    """
    global dataframes
    
    try:
        if sheet_name not in dataframes:
            return f"Error: Sheet '{sheet_name}' not found"
        
        df = dataframes[sheet_name]
        
        # Write to Excel
        df.to_excel(output_file, sheet_name=sheet_name, index=False)
        
        return f"Successfully wrote {len(df)} rows from {sheet_name} to {output_file}"
    
    except Exception as e:
        logger.error(f"Error writing results: {str(e)}")
        return f"Error: {str(e)}"

@tool
def data_validation(sheet_name: str) -> str:
    """
    Perform data validation checks.
    
    Args:
        sheet_name: Name of the worksheet to validate
        
    Returns:
        Summary of validation checks
    """
    global dataframes
    
    try:
        if sheet_name not in dataframes:
            return f"Error: Sheet '{sheet_name}' not found"
        
        df = dataframes[sheet_name]
        
        # Perform validation checks
        validation_results = []
        
        # Check for missing values
        missing_values = df.isna().sum()
        missing_cols = missing_values[missing_values > 0]
        if not missing_cols.empty:
            validation_results.append("Missing values found in columns:")
            for col, count in missing_cols.items():
                validation_results.append(f"  - {col}: {count} missing values")
        else:
            validation_results.append("No missing values found.")
        
        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            validation_results.append(f"Found {duplicate_count} duplicate rows.")
        else:
            validation_results.append("No duplicate rows found.")
        
        # Check data types
        validation_results.append("Column data types:")
        for col, dtype in df.dtypes.items():
            validation_results.append(f"  - {col}: {dtype}")
        
        return "\n".join(validation_results)
    
    except Exception as e:
        logger.error(f"Error validating data: {str(e)}")
        return f"Error: {str(e)}"

@tool
def get_column_mapping(sheet_name: str, requested_columns: List[str]) -> str:
    """
    Map requested column names to available column names.
    
    Args:
        sheet_name: Name of the worksheet
        requested_columns: List of requested column names
        
    Returns:
        Mapping of requested columns to available columns
    """
    global dataframes
    
    try:
        if sheet_name not in dataframes:
            return f"Error: Sheet '{sheet_name}' not found"
        
        df = dataframes[sheet_name]
        available_columns = list(df.columns)
        
        # Map column names
        mapping = map_column_names(requested_columns, available_columns)
        
        # Prepare result
        result = ["Column mapping:"]
        for req_col, avail_col in mapping.items():
            result.append(f"  - {req_col} -> {avail_col}")
        
        return "\n".join(result) 