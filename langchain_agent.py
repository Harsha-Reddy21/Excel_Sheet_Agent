from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from langchain.callbacks.base import BaseCallbackHandler
import pandas as pd
from typing import Dict, List, Any, Optional
import logging

# Import local modules
from excel_reader import read_excel_file, create_langchain_documents
from data_operations import (
    filter_data, filter_by_date_range, aggregate_data, 
    sort_data, pivot_table, merge_worksheets, apply_formula
)
from column_mapper import map_column_names, llm_column_mapping
from nlp_processor import (
    parse_natural_language_query, extract_date_references,
    extract_comparison_operators, identify_chart_type
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable to store dataframes
dataframes = {}

def create_excel_agent(llm):
    """Create a LangChain agent for Excel operations."""
    
    # Define tools
    tools = [
        Tool(
            name="get_available_data",
            func=lambda: get_available_data(),
            description="Get information about the currently loaded Excel data."
        ),
        Tool(
            name="filter_data",
            func=lambda sheet_name, conditions: filter_sheet_data(sheet_name, conditions),
            description="Filter data based on conditions."
        ),
        Tool(
            name="filter_by_date",
            func=lambda sheet_name, date_column, **kwargs: filter_sheet_by_date(sheet_name, date_column, **kwargs),
            description="Filter data by date range."
        ),
        Tool(
            name="aggregate_data",
            func=lambda sheet_name, group_by, aggregations: aggregate_sheet_data(sheet_name, group_by, aggregations),
            description="Aggregate data by grouping and applying aggregation functions."
        ),
        Tool(
            name="sort_data",
            func=lambda sheet_name, sort_by: sort_sheet_data(sheet_name, sort_by),
            description="Sort data by specified columns."
        ),
        Tool(
            name="create_pivot",
            func=lambda sheet_name, index, columns, values, aggfunc="mean": create_pivot_table(sheet_name, index, columns, values, aggfunc),
            description="Create a pivot table."
        ),
        Tool(
            name="parse_query",
            func=lambda query: parse_query_to_operations(llm, query),
            description="Parse a natural language query into structured operations."
        )
    ]
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an Excel data analysis assistant that helps users analyze data using natural language.
        
        Follow these steps:
        1. First, use the get_available_data tool to understand what data is available
        2. Use the appropriate tools to process the data based on the user's query
        3. Return results in a clear, concise format
        
        Be flexible with column names, handle dates intelligently, and break complex queries into steps.
        
        IMPORTANT: Do not ask for file paths or try to read files directly. The data is already loaded.
        Always start by checking what data is available using the get_available_data tool."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Create agent
    agent = create_openai_tools_agent(llm, tools, prompt)
    
    # Create agent executor
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        memory=None  # This will allow us to handle chat_history manually
    )

def set_dataframes(dfs):
    """Set the global dataframes from the uploaded file."""
    global dataframes
    dataframes = dfs

def get_available_data() -> str:
    """Get information about the currently loaded Excel data."""
    global dataframes
    
    if not dataframes:
        return "No Excel data is currently loaded. Please upload an Excel file first."
    
    # Create summary
    summary = ["Available data:"]
    for sheet, df in dataframes.items():
        summary.append(f"Sheet: {sheet}")
        summary.append(f"Rows: {len(df)}")
        summary.append(f"Columns: {', '.join(df.columns.tolist())}")
        summary.append("---")
    
    return "\n".join(summary)

def read_excel_and_store(file_path: str) -> str:
    """Read Excel file and store dataframes."""
    global dataframes
    
    try:
        # Read Excel file
        dfs = read_excel_file(file_path)
        dataframes = dfs
        
        # Create summary
        summary = []
        for sheet, df in dfs.items():
            summary.append(f"Sheet: {sheet}")
            summary.append(f"Rows: {len(df)}")
            summary.append(f"Columns: {', '.join(df.columns[:5])}...")
        
        return "\n".join(summary)
    except Exception as e:
        return f"Error reading Excel file: {str(e)}"

def filter_sheet_data(sheet_name: str, conditions: Dict[str, Any]) -> str:
    """Filter sheet data based on conditions."""
    global dataframes
    
    if sheet_name not in dataframes:
        return f"Sheet '{sheet_name}' not found"
    
    df = dataframes[sheet_name]
    filtered_df = filter_data(df, conditions)
    
    # Store result
    result_name = f"{sheet_name}_filtered"
    dataframes[result_name] = filtered_df
    
    return f"Filtered {sheet_name} from {len(df)} to {len(filtered_df)} rows"

def filter_sheet_by_date(sheet_name: str, date_column: str, 
                       start_date: Optional[str] = None, 
                       end_date: Optional[str] = None,
                       period: Optional[str] = None) -> str:
    """Filter sheet data by date range."""
    global dataframes
    
    if sheet_name not in dataframes:
        return f"Sheet '{sheet_name}' not found"
    
    df = dataframes[sheet_name]
    filtered_df = filter_by_date_range(df, date_column, start_date, end_date, period)
    
    # Store result
    result_name = f"{sheet_name}_date_filtered"
    dataframes[result_name] = filtered_df
    
    return f"Date filtered {sheet_name} from {len(df)} to {len(filtered_df)} rows"

def aggregate_sheet_data(sheet_name: str, group_by: List[str], 
                       aggregations: Dict[str, str]) -> str:
    """Aggregate sheet data."""
    global dataframes
    
    if sheet_name not in dataframes:
        return f"Sheet '{sheet_name}' not found"
    
    df = dataframes[sheet_name]
    agg_df = aggregate_data(df, group_by, aggregations)
    
    # Store result
    result_name = f"{sheet_name}_aggregated"
    dataframes[result_name] = agg_df
    
    return f"Aggregated {sheet_name} from {len(df)} to {len(agg_df)} rows"

def sort_sheet_data(sheet_name: str, sort_by: List[Any]) -> str:
    """Sort sheet data."""
    global dataframes
    
    if sheet_name not in dataframes:
        return f"Sheet '{sheet_name}' not found"
    
    df = dataframes[sheet_name]
    sorted_df = sort_data(df, sort_by)
    
    # Store result
    result_name = f"{sheet_name}_sorted"
    dataframes[result_name] = sorted_df
    
    return f"Sorted {sheet_name} data"

def create_pivot_table(sheet_name: str, index: List[str], columns: List[str], 
                     values: List[str], aggfunc: str = "mean") -> str:
    """Create a pivot table."""
    global dataframes
    
    if sheet_name not in dataframes:
        return f"Sheet '{sheet_name}' not found"
    
    df = dataframes[sheet_name]
    pivot_df = pivot_table(df, index, columns, values, aggfunc)
    
    # Store result
    result_name = f"{sheet_name}_pivot"
    dataframes[result_name] = pivot_df
    
    return f"Created pivot table with shape {pivot_df.shape}"

def parse_query_to_operations(llm, query: str) -> str:
    """Parse natural language query into operations."""
    global dataframes
    
    if not dataframes:
        return "No Excel data loaded. Please load an Excel file first."
    
    try:
        # Parse query
        operation = parse_natural_language_query(llm, query, dataframes)
        
        # Extract date references if needed
        if not operation.date_range and "date" in query.lower():
            date_info = extract_date_references(query)
            if date_info:
                operation.date_range = date_info
        
        # Extract comparison operators if needed
        if not operation.filters:
            operators = extract_comparison_operators(query)
            if operators:
                operation.filters = operators
        
        # Identify chart type if needed
        if not operation.chart_type and any(term in query.lower() for term in ["chart", "graph", "plot"]):
            chart_type = identify_chart_type(query)
            if chart_type:
                operation.chart_type = chart_type
        
        return f"Parsed query into {operation.operation_type} operation"
    
    except Exception as e:
        return f"Error parsing query: {str(e)}" 