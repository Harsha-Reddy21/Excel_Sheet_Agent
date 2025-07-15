import re
from typing import Dict, List, Any, Optional
import logging
import pandas as pd
from langchain.llms import BaseLLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain.agents import tool

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define output models for parsing
class FilterCondition(BaseModel):
    column: str = Field(description="Column name to filter on")
    operator: str = Field(description="Operator (eq, gt, lt, contains, etc.)")
    value: Any = Field(description="Value to compare against")

class DataOperation(BaseModel):
    operation_type: str = Field(description="Operation type (filter, aggregate, sort, pivot)")
    sheet_name: Optional[str] = Field(None, description="Worksheet name")
    columns: List[str] = Field(default_factory=list, description="Columns to include")
    filters: List[FilterCondition] = Field(default_factory=list, description="Filter conditions")
    group_by: List[str] = Field(default_factory=list, description="Columns to group by")
    aggregations: Dict[str, str] = Field(default_factory=dict, description="Aggregation functions")
    sort_by: List[Dict[str, str]] = Field(default_factory=list, description="Sort columns")
    pivot_index: List[str] = Field(default_factory=list, description="Pivot table index")
    pivot_columns: List[str] = Field(default_factory=list, description="Pivot table columns")
    pivot_values: List[str] = Field(default_factory=list, description="Pivot table values")
    date_column: Optional[str] = Field(None, description="Date column for filtering")
    date_range: Optional[Dict[str, Any]] = Field(None, description="Date range for filtering")
    chart_type: Optional[str] = Field(None, description="Chart type to create")

def create_query_prompt() -> PromptTemplate:
    """Create a prompt template for Excel queries."""
    template = """
    You are an AI assistant that analyzes Excel data using natural language queries.
    
    Available worksheets and their columns:
    {worksheet_info}
    
    User query: {query}
    
    Determine the operations needed to answer this query.
    Return a JSON object with the following structure:
    
    {format_instructions}
    
    Map column names correctly, choose appropriate operations, and set all relevant fields.
    """
    
    parser = PydanticOutputParser(pydantic_object=DataOperation)
    
    return PromptTemplate(
        template=template,
        input_variables=["worksheet_info", "query"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

def get_worksheet_info(dataframes: Dict[str, pd.DataFrame]) -> str:
    """Generate worksheet information string."""
    info = []
    
    for sheet_name, df in dataframes.items():
        sheet_info = f"Worksheet: {sheet_name}\n"
        sheet_info += f"Columns: {', '.join(df.columns)}\n"
        
        # Add sample values for first row
        if not df.empty:
            sheet_info += "Sample values:\n"
            for col in df.columns[:5]:  # Limit to first 5 columns
                val = str(df[col].iloc[0])[:50]  # Truncate long values
                sheet_info += f"  - {col}: {val}\n"
        
        info.append(sheet_info)
    
    return "\n".join(info)

@tool
def parse_natural_language_query(llm: BaseLLM, query: str, 
                               dataframes: Dict[str, pd.DataFrame]) -> DataOperation:
    """Parse natural language query into structured operation."""
    try:
        # Create prompt with worksheet information
        worksheet_info = get_worksheet_info(dataframes)
        prompt = create_query_prompt()
        
        # Create LLMChain
        chain = LLMChain(llm=llm, prompt=prompt)
        
        # Get response from LLM
        response = chain.run(worksheet_info=worksheet_info, query=query)
        
        # Parse response
        parser = PydanticOutputParser(pydantic_object=DataOperation)
        operation = parser.parse(response)
        
        logger.info(f"Parsed query: {query} → {operation.operation_type}")
        return operation
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        # Return a default operation if parsing fails
        return DataOperation(operation_type="filter")

@tool
def extract_date_references(query: str) -> Dict[str, Any]:
    """Extract date references from query text."""
    date_info = {}
    
    # Check for quarter patterns (e.g., "Q1 2023")
    quarter_match = re.search(r'Q([1-4])\s+(\d{4})', query, re.IGNORECASE)
    if quarter_match:
        quarter = quarter_match.group(1)
        year = quarter_match.group(2)
        date_info["period"] = f"q{quarter}_{year}"
    
    # Check for relative date terms
    date_terms = [
        (r'\blast\s+7\s+days\b', "last_7_days"),
        (r'\blast\s+30\s+days\b', "last_30_days"),
        (r'\bthis\s+month\b', "this_month"),
        (r'\blast\s+month\b', "last_month"),
        (r'\bthis\s+year\b', "this_year"),
        (r'\byesterday\b', "yesterday"),
        (r'\btoday\b', "today")
    ]
    
    for pattern, period in date_terms:
        if re.search(pattern, query, re.IGNORECASE):
            date_info["period"] = period
            break
    
    return date_info

@tool
def extract_comparison_operators(query: str) -> List[Dict[str, Any]]:
    """Extract comparison operators from query text."""
    operators = []
    
    # Patterns for numeric comparisons
    patterns = [
        (r'(\w+)\s*>\s*(\d+(?:\.\d+)?)', "gt"),
        (r'(\w+)\s*<\s*(\d+(?:\.\d+)?)', "lt"),
        (r'(\w+)\s*>=\s*(\d+(?:\.\d+)?)', "gte"),
        (r'(\w+)\s*<=\s*(\d+(?:\.\d+)?)', "lte"),
        (r'(\w+)\s*=\s*(\d+(?:\.\d+)?)', "eq"),
        (r'(\w+)\s*!=\s*(\d+(?:\.\d+)?)', "neq")
    ]
    
    for pattern, op in patterns:
        for match in re.finditer(pattern, query):
            column = match.group(1)
            value = match.group(2)
            
            # Convert value to number
            try:
                value = float(value) if '.' in value else int(value)
            except:
                pass
            
            operators.append({
                "column": column,
                "operator": op,
                "value": value
            })
    
    # Patterns for text comparisons
    text_patterns = [
        (r'(\w+)\s+contains\s+[\'"]([^\'"]+)[\'"]', "contains"),
        (r'(\w+)\s+like\s+[\'"]([^\'"]+)[\'"]', "contains")
    ]
    
    for pattern, op in text_patterns:
        for match in re.finditer(pattern, query):
            operators.append({
                "column": match.group(1),
                "operator": op,
                "value": match.group(2)
            })
    
    return operators

@tool
def identify_chart_type(query: str) -> Optional[str]:
    """Identify chart type requested in query."""
    chart_patterns = {
        "bar": [r'bar\s+chart', r'bar\s+graph', r'barchart'],
        "line": [r'line\s+chart', r'line\s+graph', r'linechart'],
        "pie": [r'pie\s+chart', r'piechart'],
        "scatter": [r'scatter\s+plot', r'scatter\s+chart', r'scatterplot'],
        "histogram": [r'histogram'],
        "heatmap": [r'heatmap', r'heat\s+map', r'correlation']
    }
    
    for chart_type, patterns in chart_patterns.items():
        if any(re.search(pattern, query, re.IGNORECASE) for pattern in patterns):
            return chart_type
    
    return None 