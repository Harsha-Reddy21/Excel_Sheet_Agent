import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
from typing import Dict, List, Optional
import logging
from langchain.agents import tool

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@tool
def create_bar_chart(df: pd.DataFrame, x_column: str, y_column: str, 
                   title: str = "") -> str:
    """Create a bar chart from dataframe columns."""
    if x_column not in df.columns or y_column not in df.columns:
        return ""
    
    try:
        plt.figure(figsize=(10, 6))
        ax = df.plot.bar(x=x_column, y=y_column, legend=False)
        
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        ax.set_title(title if title else f"{y_column} by {x_column}")
        
        plt.tight_layout()
        return convert_plot_to_base64()
    except:
        return ""

@tool
def create_line_chart(df: pd.DataFrame, x_column: str, y_columns: List[str], 
                    title: str = "") -> str:
    """Create a line chart from dataframe columns."""
    if x_column not in df.columns:
        return ""
    
    valid_y_columns = [col for col in y_columns if col in df.columns]
    if not valid_y_columns:
        return ""
    
    try:
        plt.figure(figsize=(10, 6))
        
        # Sort by x if it's a date
        if pd.api.types.is_datetime64_any_dtype(df[x_column]):
            df = df.sort_values(by=x_column)
        
        for col in valid_y_columns:
            plt.plot(df[x_column], df[col], marker='o', linestyle='-', label=col)
        
        plt.xlabel(x_column)
        plt.ylabel(', '.join(valid_y_columns))
        plt.title(title if title else "Line Chart")
        
        if len(valid_y_columns) > 1:
            plt.legend()
        
        plt.tight_layout()
        return convert_plot_to_base64()
    except:
        return ""

@tool
def create_pie_chart(df: pd.DataFrame, labels_column: str, values_column: str,
                   title: str = "") -> str:
    """Create a pie chart from dataframe columns."""
    if labels_column not in df.columns or values_column not in df.columns:
        return ""
    
    try:
        plt.figure(figsize=(8, 8))
        plt.pie(df[values_column], labels=df[labels_column], autopct='%1.1f%%',
               shadow=True, startangle=90)
        plt.axis('equal')
        plt.title(title if title else f"Distribution of {values_column}")
        
        return convert_plot_to_base64()
    except:
        return ""

@tool
def create_scatter_plot(df: pd.DataFrame, x_column: str, y_column: str, 
                      color_column: Optional[str] = None, title: str = "") -> str:
    """Create a scatter plot from dataframe columns."""
    if x_column not in df.columns or y_column not in df.columns:
        return ""
    
    try:
        plt.figure(figsize=(10, 6))
        
        if color_column and color_column in df.columns:
            scatter = plt.scatter(df[x_column], df[y_column], c=df[color_column], cmap='viridis')
            plt.colorbar(scatter, label=color_column)
        else:
            plt.scatter(df[x_column], df[y_column])
        
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.title(title if title else f"{y_column} vs {x_column}")
        
        plt.tight_layout()
        return convert_plot_to_base64()
    except:
        return ""

@tool
def create_histogram(df: pd.DataFrame, column: str, bins: int = 10, title: str = "") -> str:
    """Create a histogram from a dataframe column."""
    if column not in df.columns:
        return ""
    
    try:
        plt.figure(figsize=(10, 6))
        plt.hist(df[column].dropna(), bins=bins, color='skyblue', edgecolor='black')
        
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.title(title if title else f"Distribution of {column}")
        
        plt.tight_layout()
        return convert_plot_to_base64()
    except:
        return ""

@tool
def create_heatmap(df: pd.DataFrame, columns: Optional[List[str]] = None, title: str = "") -> str:
    """Create a correlation heatmap from dataframe columns."""
    # Select numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        return ""
    
    if columns:
        valid_columns = [col for col in columns if col in numeric_df.columns]
        if valid_columns:
            numeric_df = numeric_df[valid_columns]
    
    try:
        corr_matrix = numeric_df.corr()
        
        plt.figure(figsize=(10, 8))
        im = plt.imshow(corr_matrix, cmap='coolwarm')
        plt.colorbar(im, label='Correlation')
        
        plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45)
        plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
        
        plt.title(title if title else "Correlation Heatmap")
        plt.tight_layout()
        
        return convert_plot_to_base64()
    except:
        return ""

def convert_plot_to_base64() -> str:
    """Convert matplotlib plot to base64 string."""
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    return image_base64 