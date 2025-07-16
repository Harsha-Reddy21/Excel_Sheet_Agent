# Intelligent Excel Agent

An intelligent Excel agent built with LangChain, Streamlit, and OpenAI that processes large Excel files, understands natural language queries, and handles production scenarios including inconsistent column naming and edge cases.

## Features

- **Large File & Multi-Tab Handling**
  - Support for 10,000+ rows and multiple worksheets
  - Memory-efficient chunking strategies
  - Handles different data types and worksheet navigation

- **Natural Language Processing**
  - Integrates with OpenAI's LLM to interpret user queries
  - Generates data operations from natural language
  - Supports complex analysis (filtering, aggregations, pivoting)

- **Column Name Mapping**
  - Handles different naming conventions (snake_case, camelCase, "Proper Case")
  - Supports synonyms ("qty" vs "quantity", "amt" vs "amount")
  - Fuzzy matching algorithm for column names

- **Production Edge Cases**
  - Handles corrupted files, merged cells, memory limits
  - Manages inconsistent data types, missing values, date format inconsistencies
  - Processes ambiguous queries and non-existent columns

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd excel-sheet-agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
```

4. Edit `.env` file and add your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and navigate to the provided URL (usually http://localhost:8501)

3. Enter your OpenAI API key in the sidebar

4. Upload an Excel file

5. Start asking questions about your data!

## Example Queries

- "Show sales data for Q3 2024 where revenue > 50000"
- "Create pivot table showing total sales by region and product"
- "Find customers who haven't ordered in 6 months"
- "What are the top 5 products by sales volume?"
- "Show me a summary of revenue by month"

## Project Structure

- `app.py`: Streamlit interface for the Excel agent
- `excel_processor.py`: Core Excel processing functions
- `langchain_tools.py`: LangChain tools for Excel operations
- `requirements.txt`: Required Python packages

## Technical Details

- Handles files up to 100MB
- Processes queries within 10 seconds
- Supports concurrent users
- Implements security measures (input validation, file scanning)

## License

MIT

## Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain) for the agent framework
- [Streamlit](https://streamlit.io/) for the web interface
- [OpenAI](https://openai.com/) for the language model
- [Pandas](https://pandas.pydata.org/) for data processing 