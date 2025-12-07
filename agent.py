# agent.py
import asyncio
import os
import sys
import streamlit as st # Added Streamlit for UI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
# FIX: Use the correct, modern imports for Agent components
from langchain.agents import create_agent
from langchain.agents import AgentExecutor 
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

# --- 1. CONFIGURATION ---
# These variables MUST be set in your Heroku Config Vars:
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
JAPAN_PARTS_SERVER_URL = os.environ.get("JAPAN_PARTS_SERVER_URL")

GROQ_MODEL = "llama-3.3-70b-versatile"

# --- Error Check (For Heroku startup, Streamlit handles some errors gracefully) ---
if not GROQ_API_KEY:
    # In a Streamlit app, display the error instead of crashing the whole process
    st.error("GROQ_API_KEY environment variable not set. Please configure Heroku Config Vars.")
    sys.exit(1)

# --- Agent Logic ---
async def initialize_agent():
    """Initializes the LLM and dynamically loads tools from the MCP server."""
    
    # 2. Initialize the MultiServerMCPClient
    if not JAPAN_PARTS_SERVER_URL:
        st.warning("JAPAN_PARTS_SERVER_URL not set. Agent will run without access to the parts database.")
        mcp_tools = []
    else:
        mcp_client = MultiServerMCPClient(
            servers={
                "japan_parts_db": {
                    "url": JAPAN_PARTS_SERVER_URL,
                    "transport": "streamable-http"
                }
            }
        )

        # 3. Load MCP Tools and Translate for the LLM
        try:
            mcp_tools = await load_mcp_tools(mcp_client)
            if mcp_tools:
                st.success(f"✅ Connected to MCP server. Loaded {len(mcp_tools)} tool(s).")
            else:
                st.warning("Connected to MCP server but no tools were found.")
        except Exception as e:
            st.error(f"⚠️ Failed to load tools from MCP server. Error: {e}")
            mcp_tools = []

    # 4. Configure the LLM
    llm = ChatGroq(model=GROQ_MODEL, temperature=0, api_key=GROQ_API_KEY)
    
    # CRITICAL: The system prompt for tool use enforcement.
    system_prompt = (
        "You are an expert parts agent powered by Groq and Llama 3. Your role is to "
        "provide accurate, real-time data on Japan parts pricing, stock, or "
        "specifications. You MUST use the available tool(s) for any query about "
        "specific parts, stock, or price, as your internal knowledge is outdated. "
        "Do not answer these questions without the tool output."
    )
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    # 5. Create the Agent and Executor
    agent = create_agent(llm, mcp_tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=mcp_tools, verbose=True)
    
    return agent_executor

async def handle_user_input(user_query, agent_executor):
    """Runs the agent with the user's query."""
    try:
        # Run the Agent (Handles the Full Tool-Calling Loop)
        with st.spinner('Thinking... (using Groq and MCP tool, this will be fast!)'):
            result = await agent_executor.ainvoke({"input": user_query})
            return result['output']
    except Exception as e:
        return f"An error occurred while processing your request: {e}"

# --- Streamlit UI Implementation ---

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="MCP Groq Parts Agent", layout="wide")
    st.title("🇯🇵 Groq Agent with MCP Parts Database")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Initialize the agent executor once
    if "agent_executor" not in st.session_state:
        # Streamlit doesn't support direct asyncio.run in the main flow, 
        # so we run it once here to set up the async object.
        st.session_state.agent_executor = asyncio.run(initialize_agent())
        
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask about part prices, stock, or specs..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get the agent's response
        agent_executor = st.session_state.agent_executor
        
        # Run the async handler
        response = asyncio.run(handle_user_input(prompt, agent_executor))
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
