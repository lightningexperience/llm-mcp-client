# agent.py (version 1.04) — Using version-stable create_tool_calling_agent
import asyncio
import os
import sys
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# ✨ This import ALWAYS exists in LangChain 0.2.x — safe!
from langchain.agents import create_tool_calling_agent

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

# --- 1. CONFIGURATION ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
JAPAN_PARTS_SERVER_URL = os.environ.get("JAPAN_PARTS_SERVER_URL")

GROQ_MODEL = "llama-3.3-70b-versatile"

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is not set in Heroku Config Vars.")
    sys.exit(1)

# --- Agent Initialization ---
async def initialize_agent():
    """Initializes LLM and loads MCP tools if available."""

    # Load MCP tools
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
            mcp_tools = await load_mcp_tools(client)
            st.success(f"Loaded {len(mcp_tools)} MCP tool(s).")
        except Exception as e:
            st.error(f"MCP load error: {e}")
            mcp_tools = []

    # Configure LLM
    llm = ChatGroq(model=GROQ_MODEL, temperature=0, api_key=GROQ_API_KEY)

    # Prompt
    system_prompt = (
        "You are an expert Japan auto parts agent using Groq Llama 3. "
        "For any question involving part price, availability, or specs, "
        "you MUST use the provided tool rather than guessing."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ]
    )

    # --- ⭐ SAFEST LC AGENT BUILDER AVAILABLE ⭐ ---
    agent = create_tool_calling_agent(
        llm=llm,
        tools=mcp_tools,
        prompt=prompt
    )

    # No AgentExecutor wrapper — agent itself is runnable
    return agent


async def handle_user_input(query, agent):
    try:
        with st.spinner("Thinking..."):
            result = await agent.ainvoke({"input": query})
            return result["output"]
    except Exception as e:
        return f"Error during agent run: {e}"


# --- Streamlit UI ---
def main():
    st.set_page_config(page_title="Groq MCP Parts Agent", layout="wide")
    st.title("🇯🇵 Groq-Llama + MCP Parts Lookup Agent")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "agent" not in st.session_state:
        st.session_state.agent = asyncio.run(initialize_agent())

    # Display history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User Message Input
    if prompt := st.chat_input("Ask about part prices, stock, specs..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        response = asyncio.run(handle_user_input(prompt, st.session_state.agent))

        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
