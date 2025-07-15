import streamlit as st
import pandas as pd
import os
import tempfile
import logging
from dotenv import load_dotenv

# Import LangChain components
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StreamlitCallbackHandler
from langchain.schema import SystemMessage, HumanMessage, AIMessage

# Import local modules
from excel_reader import read_excel_file
from langchain_agent import create_excel_agent, set_dataframes

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_session():
    """Initialize session state variables."""
    if 'dataframes' not in st.session_state:
        st.session_state.dataframes = {}
    if 'current_sheet' not in st.session_state:
        st.session_state.current_sheet = None
    if 'file_uploaded' not in st.session_state:
        st.session_state.file_uploaded = False
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'agent' not in st.session_state:
        st.session_state.agent = None

def setup_llm():
    """Set up LLM based on user selection."""
    model_option = st.sidebar.selectbox(
        "Select LLM Model",
        ["GPT-3.5", "GPT-4", "Claude (if available)"]
    )
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    if model_option == "GPT-4":
        return ChatOpenAI(temperature=0, model="gpt-4", api_key=api_key)
    elif model_option == "Claude (if available)":
        try:
            from langchain.chat_models import ChatAnthropic
            anthropic_key = os.getenv("ANTHROPIC_API_KEY")
            if anthropic_key:
                return ChatAnthropic(temperature=0, api_key=anthropic_key)
        except:
            st.sidebar.warning("Claude not available. Using GPT-3.5 instead.")
    
    # Default to GPT-3.5
    return ChatOpenAI(temperature=0, model="gpt-3.5-turbo", api_key=api_key)

def handle_file_upload():
    """Handle Excel file upload."""
    uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=["xlsx", "xls"])
    
    if uploaded_file:
        try:
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
                tmp.write(uploaded_file.getvalue())
                temp_path = tmp.name
            
            # Read Excel file
            dataframes = read_excel_file(temp_path)
            st.session_state.dataframes = dataframes
            st.session_state.file_uploaded = True
            
            # Set current sheet
            if dataframes:
                first_sheet = list(dataframes.keys())[0]
                st.session_state.current_sheet = first_sheet
            
            # Clean up temp file
            os.unlink(temp_path)
            
            # Create sheet selector
            sheet_names = list(dataframes.keys())
            st.session_state.current_sheet = st.sidebar.selectbox(
                "Select Sheet", sheet_names, index=0
            )
            
            # Success message
            st.sidebar.success(f"File uploaded with {len(dataframes)} sheets")
            
            # Initialize agent with dataframes
            llm = setup_llm()
            st.session_state.agent = create_excel_agent(llm)
            
            # Set dataframes in the agent
            set_dataframes(dataframes)
            
        except Exception as e:
            st.sidebar.error(f"Error: {str(e)}")
    
    return st.session_state.file_uploaded

def display_data_preview():
    """Display preview of current dataframe."""
    if not st.session_state.current_sheet:
        return
    
    df = st.session_state.dataframes[st.session_state.current_sheet]
    
    # Basic info
    st.subheader(f"Sheet: {st.session_state.current_sheet}")
    col1, col2 = st.columns(2)
    col1.metric("Rows", len(df))
    col2.metric("Columns", len(df.columns))
    
    # Sample data
    with st.expander("Preview Data"):
        st.dataframe(df.head(5))
    
    # Column info
    with st.expander("Column Information"):
        col_info = []
        for col in df.columns:
            col_info.append({
                "Column": col,
                "Type": str(df[col].dtype),
                "Missing": df[col].isna().sum()
            })
        st.table(pd.DataFrame(col_info))

def process_query(query):
    """Process natural language query using agent."""
    if not st.session_state.file_uploaded:
        st.error("Please upload an Excel file first")
        return
    
    if not st.session_state.agent:
        llm = setup_llm()
        st.session_state.agent = create_excel_agent(llm)
        set_dataframes(st.session_state.dataframes)
    
    # Add query to history
    st.session_state.query_history.append(query)
    
    # Add to chat history
    st.session_state.chat_history.append(HumanMessage(content=query))
    
    # Process query
    with st.container():
        st.write("Processing query...")
        st_callback = StreamlitCallbackHandler(st.container())
        
        try:
            response = st.session_state.agent.invoke(
                {
                    "input": query,
                    "chat_history": st.session_state.chat_history
                },
                {"callbacks": [st_callback]}
            )
            
            # Display result
            st.write("Result:")
            st.write(response["output"])
            
            # Add AI response to chat history
            st.session_state.chat_history.append(AIMessage(content=response["output"]))
            
            # Check if result is a dataframe
            if isinstance(response.get("output"), pd.DataFrame):
                st.dataframe(response["output"])
        except Exception as e:
            st.error(f"Error: {str(e)}")

def main():
    """Main application function."""
    # Page config
    st.set_page_config(
        page_title="Excel Sheet Agent",
        page_icon="📊",
        layout="wide"
    )
    
    # Initialize session
    initialize_session()
    
    # Header
    st.title("Excel Sheet Agent")
    st.markdown("Analyze Excel data using natural language queries")
    
    # Sidebar
    st.sidebar.title("Settings")
    
    # File upload
    file_uploaded = handle_file_upload()
    
    # Main content
    if file_uploaded:
        # Display data preview
        display_data_preview()
        
        # Query input
        st.subheader("Ask about your data")
        query = st.text_area("Enter your query:", height=100,
                          placeholder="Example: Show me a summary of the data")
        
        # Process query
        if st.button("Process Query") or query:
            if query:
                process_query(query)
        
        # Query history
        if st.session_state.query_history:
            with st.expander("Query History"):
                for i, q in enumerate(st.session_state.query_history):
                    st.write(f"{i+1}. {q}")
    else:
        # Welcome screen
        st.info("Please upload an Excel file to get started")
        
        # Features
        st.subheader("Features")
        st.markdown("""
        - Process large Excel files (10,000+ rows)
        - Handle multiple worksheets
        - Ask questions in plain English
        - Perform complex analysis (filtering, aggregations, pivoting)
        - Handle different naming conventions
        """)

if __name__ == "__main__":
    main() 