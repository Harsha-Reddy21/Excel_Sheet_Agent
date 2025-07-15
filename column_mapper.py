from typing import Dict, List, Optional
from fuzzywuzzy import fuzz
import logging
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dictionary of common business term synonyms (shortened)
SYNONYM_DICT = {
    "quantity": ["qty", "quant", "amount", "count", "num"],
    "revenue": ["rev", "sales", "income", "earnings"],
    "customer": ["cust", "client", "buyer", "purchaser"],
    "product": ["prod", "item", "merchandise", "goods"],
    "date": ["dt", "day", "time", "timestamp"],
    "price": ["cost", "amount", "value", "charge"],
    "region": ["area", "territory", "zone", "sector"]
}

def normalize_column_name(column: str) -> str:
    """Normalize column name to lowercase with underscores."""
    import re
    normalized = column.lower()
    normalized = re.sub(r'[^a-z0-9]', '_', normalized)
    return re.sub(r'_+', '_', normalized).strip('_')

def get_column_similarity(col1: str, col2: str) -> float:
    """Calculate similarity score between column names."""
    norm_col1 = normalize_column_name(col1)
    norm_col2 = normalize_column_name(col2)
    
    return max(
        fuzz.ratio(norm_col1, norm_col2),
        fuzz.partial_ratio(norm_col1, norm_col2),
        fuzz.token_sort_ratio(norm_col1, norm_col2)
    )

def find_best_column_match(query_column: str, available_columns: List[str], 
                         threshold: int = 70) -> Optional[str]:
    """Find best matching column from available columns."""
    best_match = None
    best_score = 0
    
    for col in available_columns:
        score = get_column_similarity(query_column, col)
        if score > best_score:
            best_score = score
            best_match = col
    
    return best_match if best_score >= threshold else None

def check_synonym_match(query_column: str, available_columns: List[str]) -> Optional[str]:
    """Check if query column has a synonym in available columns."""
    norm_query = normalize_column_name(query_column)
    
    # Check if query is a key in synonym dictionary
    for key, synonyms in SYNONYM_DICT.items():
        if norm_query == normalize_column_name(key):
            for col in available_columns:
                if any(normalize_column_name(syn) == normalize_column_name(col) for syn in synonyms):
                    return col
        
        # Check if query is a synonym
        if any(normalize_column_name(syn) == norm_query for syn in synonyms):
            for col in available_columns:
                norm_col = normalize_column_name(col)
                if (norm_col == normalize_column_name(key) or 
                    any(normalize_column_name(syn) == norm_col for syn in synonyms)):
                    return col
    
    return None

def map_column_names(requested_columns: List[str], available_columns: List[str]) -> Dict[str, str]:
    """Map requested column names to available column names."""
    column_mapping = {}
    
    for req_col in requested_columns:
        # Check for exact match
        if req_col in available_columns:
            column_mapping[req_col] = req_col
        # Check for synonym match
        elif syn_match := check_synonym_match(req_col, available_columns):
            column_mapping[req_col] = syn_match
        # Try fuzzy matching
        elif fuzzy_match := find_best_column_match(req_col, available_columns):
            column_mapping[req_col] = fuzzy_match
    
    return column_mapping

def create_column_mapping_chain(llm):
    """Create a LangChain for column mapping."""
    template = """
    I need to map requested column names to available column names in an Excel sheet.
    
    Requested columns: {requested_columns}
    Available columns: {available_columns}
    
    Return a JSON dictionary mapping each requested column to the most appropriate available column.
    If no match is found for a column, exclude it from the results.
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["requested_columns", "available_columns"]
    )
    
    return LLMChain(llm=llm, prompt=prompt)

def llm_column_mapping(llm, requested_columns: List[str], 
                     available_columns: List[str]) -> Dict[str, str]:
    """Use LLM to suggest column mappings."""
    try:
        chain = create_column_mapping_chain(llm)
        response = chain.run(
            requested_columns=", ".join(requested_columns),
            available_columns=", ".join(available_columns)
        )
        
        # Parse the response to get the mapping
        import json
        import re
        
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            mapping = json.loads(json_match.group(0))
            return {k: v for k, v in mapping.items() if v in available_columns}
        return {}
            
    except Exception as e:
        logger.error(f"Error using LLM for column mapping: {str(e)}")
        return {} 