# Intelligent Excel Agent

An AI-powered Excel analysis tool that can process any Excel file and respond to natural language queries about the data.

## Features

- Upload any Excel file for instant analysis
- Ask questions about your data in plain English
- Automatic detection of column types and purposes
- Smart mapping of inconsistent column names
- Comprehensive data analysis and visualization suggestions
- Works with multi-sheet Excel files and detects relationships between sheets

## Technology Stack

- **Streamlit**: Web interface
- **LangChain**: Agent framework
- **OpenAI**: Natural language processing
- **Pandas**: Data manipulation

## Installation

1. Clone this repository
```bash
git clone <repository-url>
cd Excel_Sheet_Agent
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your OpenAI API key
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Start the Streamlit app
```bash
streamlit run app.py
```

2. Upload your Excel file using the sidebar
3. Ask questions about your data in the chat interface
4. Try the example queries for quick insights

## Example Queries

- "What can you tell me about this Excel file?"
- "Summarize the data in Sheet1"
- "Find the top 5 values in the Revenue column"
- "Create a summary of numeric values grouped by categories"
- "Identify and handle missing values in the data"
- "What are the relationships between different sheets?"
- "Merge two sheets based on a common column"

## Project Structure

- `app.py`: Streamlit web interface
- `langchain_tools.py`: LangChain tools for Excel operations
- `excel_processor.py`: Core Excel processing functions

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 