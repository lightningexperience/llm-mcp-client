import asyncio
import os
import sys
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# 1. Import the Agent Constructor
from langchain.agents import create_tool_calling_agent

# 2. Import the Executor (REQUIRED to actually run the tools)
# This import works in langchain>=0.2.0 if langchain-community is installed
from langchain.agents import AgentExecutor

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

# --- CONFIGURATION ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
JAPAN_PARTS_SERVER_URL = os.environ.get("JAPAN_PARTS_SERVER_URL")
GROQ_MODEL = "llama-3.3-70b-versatile"

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is not set in Heroku Config Vars.")
    sys.exit(1)

async def initialize_agent():
    """Initializes LLM, loads tools, and creates the Agent Executor."""
    
    # --- Load MCP Tools ---
    if not JAPAN_PARTS_SERVER_URL:
        st.warning("JAPAN_PARTS_SERVER_URL missing — MCP tools disabled.")
        mcp_tools = []
    else:
        try:
            client = MultiServerMCPClient(
                servers={
                    "japan_parts_db": {
                        "url": JAPAN_PARTS_SERVER_URL,
                        "transport": "streamable-http"
                    }
                }
            )
            # Load and verify tools
            mcp_tools = await load_mcp_tools(client)
            st.success(f"✅ Loaded {len(mcp_tools)} MCP tool(s).")
        except Exception as e:
            st.error(f"MCP load error: {e}")
            mcp_tools = []

    # --- Configure LLM ---
    llm = ChatGroq(model=GROQ_MODEL, temperature=0, api_key=GROQ_API_KEY)

    # --- Define Prompt ---
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert Japan auto parts agent. You MUST use the provided tools for prices/stock."),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"), # Required for tool_calling_agent
        ]
    )

    # --- Create Agent & Executor ---
    # 1. Create the Agent (The Brain)
    agent = create_tool_calling_agent(llm, mcp_tools, prompt)

    # 2. Create the Executor (The Body - runs the loop)
    # This is what you actually call with .invoke()
    agent_executor = AgentExecutor(agent=agent, tools=mcp_tools, verbose=True)

    return agent_executor

async def handle_user_input(query, agent_executor):
    try:
        with st.spinner("Thinking..."):
            # We invoke the EXECUTOR, not the agent directly
            result = await agent_executor.ainvoke({"input": query})
            return result["output"]
    except Exception as e:
        return f"Error during agent run: {e}"

# --- Streamlit UI ---
def main():
    st.set_page_config(page_title="Groq MCP Parts Agent", layout="wide")
    st.title("🇯🇵 Groq-Llama + MCP Parts Lookup Agent")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "agent_executor" not in st.session_state:
        # Run init once and store the EXECUTOR
        st.session_state.agent_executor = asyncio.run(initialize_agent())

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about part prices, stock, specs..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Pass the executor to the handler
        response = asyncio.run(handle_user_input(prompt, st.session_state.agent_executor))

        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
