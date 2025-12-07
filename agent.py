 # agent.py (version 1.03) — Updated with correct create_react_agent import path
import asyncio
import os
import sys
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Correct modern LangChain import (working for LC 0.2.9)
from langchain.agents.react.agent import create_react_agent

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

# --- 1. CONFIGURATION ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
JAPAN_PARTS_SERVER_URL = os.environ.get("JAPAN_PARTS_SERVER_URL")

GROQ_MODEL = "llama-3.3-70b-versatile"

# --- Error Check ---
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY environment variable not set. Please configure Heroku Config Vars.")
    sys.exit(1)

# --- Agent Logic ---
async def initialize_agent():
    """Initializes the LLM and dynamically loads tools from the MCP server."""

    # 2. Initialize MultiServerMCPClient
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

        # 3. Load MCP Tools
        try:
            mcp_tools = await load_mcp_tools(mcp_client)
            if mcp_tools:
                st.success(f"✅ Connected to MCP server. Loaded {len(mcp_tools)} tool(s).")
            else:
                st.warning("Connected to MCP server but no tools were found.")
        except Exception as e:
            st.error(f"⚠️ Failed to load tools from MCP server. Error: {e}")
            mcp_tools = []

    # 4. Configure LLM
    llm = ChatGroq(model=GROQ_MODEL, temperature=0, api_key=GROQ_API_KEY)

    # System prompt for safety + tool enforcement
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

    # --- Create ReAct agent (new LC runtime uses this directly as runnable) ---
    agent = create_react_agent(
        llm=llm,
        tools=mcp_tools,
        prompt=prompt
    )

    agent_executor = agent  # No AgentExecutor wrapper anymore
    return agent_executor


async def handle_user_input(user_query, agent_executor):
    """Runs the agent with the user's query."""
    try:
        with st.spinner("Thinking... (Groq + MCP tool)"):
            result = await agent_executor.ainvoke({"input": user_query})
            return result["output"]
    except Exception as e:
        return f"An error occurred while processing your request: {e}"


# --- Streamlit UI ---
def main():
    st.set_page_config(page_title="MCP Groq Parts Agent", layout="wide")
    st.title("🇯🇵 Groq Agent with MCP Parts Database")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "agent_executor" not in st.session_state:
        st.session_state.agent_executor = asyncio.run(initialize_agent())

    # Render chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about part prices, stock, or specs..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        agent_executor = st.session_state.agent_executor
        response = asyncio.run(handle_user_input(prompt, agent_executor))

        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
