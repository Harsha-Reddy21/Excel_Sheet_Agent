import pandas as pd
import os
from typing import Dict, List, Optional
import logging
from langchain.document_loaders import DataFrameLoader

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_excel_file(file_path: str) -> Dict[str, pd.DataFrame]:
    """Read Excel file and return dictionary of dataframes."""
    try:
        excel_file = pd.ExcelFile(file_path)
        return {sheet: pd.read_excel(excel_file, sheet_name=sheet) 
                for sheet in excel_file.sheet_names}
    except Exception as e:
        logger.error(f"Error reading Excel file: {str(e)}")
        raise

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean dataframe by handling common issues."""
    df = df.dropna(how='all')
    df.columns = [str(col).strip() for col in df.columns]
    return df

def convert_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert date-like columns to datetime."""
    for col in df.columns:
        if any(date_term in col.lower() for date_term in ["date", "time", "day"]):
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except:
                pass
    return df

def create_langchain_documents(df: pd.DataFrame, sheet_name: str) -> List:
    """Convert dataframe to LangChain documents for processing."""
    loader = DataFrameLoader(df, page_content_column=df.columns[0])
    documents = loader.load()
    
    # Add metadata about sheet name
    for doc in documents:
        doc.metadata["sheet_name"] = sheet_name
        
    return documents

def chunk_dataframe(df: pd.DataFrame, chunk_size: int = 1000) -> List[pd.DataFrame]:
    """Split dataframe into smaller chunks."""
    return [df.iloc[i:i + chunk_size].copy() 
            for i in range(0, len(df), chunk_size)]

def get_dataframe_info(df: pd.DataFrame) -> Dict:
    """Get basic information about dataframe."""
    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
        "sample_rows": df.head(3).to_dict(orient="records")
    } 