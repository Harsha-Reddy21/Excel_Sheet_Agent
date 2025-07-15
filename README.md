# Excel Sheet Agent

An intelligent agent built with LangChain that processes Excel files and understands natural language queries.

## Features

- Process large Excel files (10,000+ rows)
- Support for multiple worksheets
- Natural language query processing
- Intelligent column name mapping with fuzzy matching
- Data filtering, aggregation, and pivoting
- Visualization capabilities
- Streamlit user interface

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Excel_Sheet_Agent.git
cd Excel_Sheet_Agent

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the root directory with your OpenAI API key:

```
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

Then:
1. Upload your Excel file
2. Ask questions in natural language
3. View the results and visualizations

## Example Queries

- "Show me the total sales by region"
- "What was the average revenue in Q3?"
- "Create a bar chart of top 5 products by profit"
- "Filter data where sales are greater than 10000"
- "Pivot the data to show product categories by quarter"

## Project Structure

- `app.py`: Streamlit interface
- `excel_reader.py`: Functions for reading and processing Excel files
- `column_mapper.py`: Intelligent column name mapping
- `data_operations.py`: Data filtering, aggregation, and pivoting
- `nlp_processor.py`: Natural language query processing
- `visualization.py`: Chart generation functions
- `langchain_tools.py`: LangChain tool definitions

## Requirements

See `requirements.txt` for a complete list of dependencies.

## License

MIT 