import streamlit as st
import pandas as pd
import os
import tempfile
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import StreamlitCallbackHandler
from langchain_tools import excel_tools, excel_state
import json
import time
import re

# Set page configuration
st.set_page_config(
    page_title="Excel Agent",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False
if "processing" not in st.session_state:
    st.session_state.processing = False
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""
if "temp_file_path" not in st.session_state:
    st.session_state.temp_file_path = None
if "file_analysis" not in st.session_state:
    st.session_state.file_analysis = None

# Function to initialize the agent
def initialize_excel_agent():
    if not st.session_state.openai_api_key:
        st.error("Please enter your OpenAI API key.")
        return None
    
    try:
        # Initialize the language model
        llm = ChatOpenAI(
            model_name="gpt-4",
            temperature=0,
            streaming=True,
            openai_api_key=st.session_state.openai_api_key
        )
        
        # Initialize conversation memory
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        # Initialize the agent with a system message that helps it work with any Excel file
        system_message = """You are an intelligent Excel agent that can analyze and manipulate any Excel file.
        You can handle various data structures, column naming conventions, and content types.
        When processing user queries:
        1. First understand the structure of the data
        2. Map user's terminology to the actual column names in the file
        3. Suggest relevant operations based on the data structure
        4. Provide clear explanations of your findings
        
        Remember that you don't know what's in the Excel file until you analyze it, so avoid making assumptions about specific columns or data.
        The Excel file has already been loaded for you - do not try to load a file by name.
        """
        
        # Initialize the agent
        agent = initialize_agent(
            tools=excel_tools,
            llm=llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            memory=memory,
            verbose=True,
            handle_parsing_errors=True,
            agent_kwargs={"system_message": system_message}
        )
        
        return agent
    except Exception as e:
        st.error(f"Error initializing agent: {str(e)}")
        return None

# Function to process user query
def process_query(query):
    st.session_state.processing = True
    
    if st.session_state.conversation is None:
        st.session_state.conversation = initialize_excel_agent()
        if st.session_state.conversation is None:
            st.session_state.processing = False
            return
    
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": query})
    
    # Create a placeholder for the assistant's response
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response_placeholder = st.empty()
        
        try:
            # If this is the first query and we haven't analyzed the file yet, prepend instructions
            if len(st.session_state.chat_history) <= 2 and st.session_state.temp_file_path:
                # Remind the agent that the file is already loaded
                augmented_query = f"The Excel file has already been loaded at path {st.session_state.temp_file_path}. " + query
            else:
                augmented_query = query
                
            # Run the agent
            response = st.session_state.conversation.run(augmented_query, callbacks=[st_callback])
            
            # Update the placeholder with the full response
            response_placeholder.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        except Exception as e:
            response_placeholder.error(f"Error: {str(e)}")
            st.session_state.chat_history.append({"role": "assistant", "content": f"Error: {str(e)}"})
    
    st.session_state.processing = False

# Function to handle file upload
def handle_file_upload(uploaded_file):
    if uploaded_file is not None:
        try:
            # Create a temporary file
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            
            # Save the uploaded file
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Store the file path
            st.session_state.temp_file_path = temp_path
            st.session_state.file_uploaded = True
            
            # Load the file using our tools
            if st.session_state.conversation is None:
                st.session_state.conversation = initialize_excel_agent()
            
            # Run the load_excel_file tool directly
            if st.session_state.conversation:
                # Explicitly load the file using the tool
                result = excel_tools[0](temp_path)
                
                # Parse the result to get sheet info
                try:
                    sheet_info = json.loads(result)
                    
                    # Run analysis on the Excel structure
                    analysis_result = excel_tools[-2]()  # analyze_excel_structure tool
                    try:
                        st.session_state.file_analysis = json.loads(analysis_result)
                    except:
                        st.session_state.file_analysis = None
                    
                    return sheet_info
                except json.JSONDecodeError:
                    st.error(f"Error parsing sheet info: {result}")
                    return None
            else:
                st.error("Could not initialize the conversation agent.")
                return None
            
        except Exception as e:
            st.error(f"Error uploading file: {str(e)}")
            return None
    return None

# Function to display chat messages
def display_chat_history():
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Function to generate dynamic example queries based on file content
def generate_example_queries():
    if not st.session_state.file_analysis:
        return [
            "What can you tell me about this Excel file?",
            "Analyze the structure of this data",
            "What operations would you suggest for this data?",
            "Show me a summary of the data",
            "What are the relationships between sheets?"
        ]
    
    queries = []
    
    # Add generic queries
    queries.append("Analyze this Excel file and tell me what insights you can find")
    
    # Check if we have sheets with data
    if st.session_state.file_analysis:
        sheet_names = list(st.session_state.file_analysis.keys())
        sheet_names = [name for name in sheet_names if name != "potential_relationships"]
        
        if sheet_names:
            # Get the first sheet with the most rows
            sheet_name = max(sheet_names, key=lambda x: st.session_state.file_analysis[x]["rows"])
            
            # Check if there are date columns
            if st.session_state.file_analysis[sheet_name]["column_types"]["datetime"] > 0:
                queries.append(f"Show me trends over time in the {sheet_name} sheet")
            
            # Check if there are numeric and categorical columns
            if (st.session_state.file_analysis[sheet_name]["column_types"]["numeric"] > 0 and 
                st.session_state.file_analysis[sheet_name]["column_types"]["categorical"] > 0):
                queries.append(f"Create a summary of numeric values grouped by categories in {sheet_name}")
            
            # Check for missing values
            if st.session_state.file_analysis[sheet_name]["missing_values"] > 0:
                queries.append("Identify and handle missing values in the data")
            
            # Check for potential relationships
            if "potential_relationships" in st.session_state.file_analysis and st.session_state.file_analysis["potential_relationships"]:
                rel = st.session_state.file_analysis["potential_relationships"][0]
                queries.append(f"Merge the {rel['sheet1']} and {rel['sheet2']} sheets on {rel['common_column']}")
    
    # Add more generic queries if we don't have enough
    while len(queries) < 5:
        queries.append("What insights can you provide about this data?")
        queries.append("Create a summary of the most important columns")
        queries.append("What patterns do you see in this data?")
        queries.append("How would you visualize this data?")
        queries.append("Suggest ways to clean and prepare this data")
        
        if len(queries) >= 5:
            break
    
    return queries[:5]  # Return at most 5 queries

# Main app layout
st.title("🔍 Intelligent Excel Agent")

# Sidebar for API key and file upload
with st.sidebar:
    st.header("Configuration")
    
    # API key input
    api_key = st.text_input("Enter your OpenAI API key:", type="password")
    if api_key:
        st.session_state.openai_api_key = api_key
    
    st.header("Upload Excel File")
    uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])
    
    if uploaded_file is not None:
        sheet_info = handle_file_upload(uploaded_file)
        
        if sheet_info:
            st.success(f"File uploaded: {uploaded_file.name}")
            
            st.subheader("Sheet Information")
            for sheet in sheet_info:
                with st.expander(f"{sheet['name']} ({sheet['rows']} rows)"):
                    st.write(f"Columns: {', '.join([str(c) for c in sheet['columns'] if c != '...'])}")
                    if "column_purposes" in sheet:
                        st.write("Column purposes:")
                        purposes = sheet["column_purposes"]
                        for purpose, cols in purposes.items():
                            st.write(f"- {purpose.title()}: {cols}")
    
    st.header("Example Queries")
    example_queries = generate_example_queries()
    
    for query in example_queries:
        if st.button(query):
            if not st.session_state.file_uploaded:
                st.error("Please upload an Excel file first.")
            else:
                process_query(query)

# Main chat interface
st.header("Chat with your Excel Data")

# Display chat history
display_chat_history()

# Input for new query
if user_query := st.chat_input("Ask a question about your Excel data...", disabled=not st.session_state.file_uploaded or st.session_state.processing):
    process_query(user_query)

# Instructions if no file uploaded
if not st.session_state.file_uploaded:
    st.info("👈 Please upload an Excel file from the sidebar to get started.")
    
    st.markdown("""
    ### What can this Excel Agent do?
    
    - Process Excel files of any structure and content
    - Automatically detect column types and purposes
    - Understand natural language queries about your data
    - Perform complex operations like filtering, aggregation, and pivoting
    - Create visualizations and summaries
    - Handle inconsistent column naming through intelligent mapping
    - Suggest relevant operations based on your data structure
    
    ### Example queries you can try:
    
    - "What insights can you find in this data?"
    - "Create a summary of the most important columns"
    - "Find patterns or trends in the data"
    - "Identify and handle missing values"
    - "Merge related sheets based on common columns"
    """)

# Footer
st.markdown("---")
st.caption("Powered by LangChain + OpenAI + Streamlit") 