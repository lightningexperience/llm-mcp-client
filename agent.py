# agent.py - MCP-powered Groq Chat Agent with LangChain (Heroku-ready)

import os
import sys
import asyncio
import logging
import streamlit as st
from typing import List, Dict, Any

# LangChain imports
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import AgentExecutor, create_tool_calling_agent

# MCP imports
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
JAPAN_PARTS_SERVER_URL = os.environ.get("JAPAN_PARTS_SERVER_URL")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

logger = logging.getLogger("MCPChatAgent")
logging.basicConfig(level=logging.INFO)

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY is not set in Heroku Config Vars.")
    sys.exit(1)

# -----------------------------------------------------------------------------
# Initialize Agent with MCP Tools
# -----------------------------------------------------------------------------
async def initialize_agent():
    """Initialize LangChain agent with MCP tools"""
    
    # Load MCP tools
    mcp_tools = []
    if not JAPAN_PARTS_SERVER_URL:
        st.warning("⚠️ JAPAN_PARTS_SERVER_URL not set - MCP tools disabled")
    else:
        try:
            # Create MCP client
            client = MultiServerMCPClient(
                servers={
                    "japan_parts_db": {
                        "url": JAPAN_PARTS_SERVER_URL,
                        "transport": "streamable-http"
                    }
                }
            )
            
            # Load tools from MCP server
            mcp_tools = await load_mcp_tools(client)
            logger.info(f"✅ Loaded {len(mcp_tools)} MCP tool(s)")
            
            for tool in mcp_tools:
                logger.info(f"  📦 {tool.name}: {tool.description}")
                
        except Exception as e:
            logger.error(f"❌ MCP initialization error: {e}")
            st.error(f"MCP connection failed: {e}")
    
    # Initialize Groq LLM
    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0.3,
        api_key=GROQ_API_KEY
    )
    
    # Create prompt template
    system_prompt = """You are a helpful assistant for Japan HQ auto parts inventory.

You have access to MCP tools that can search the live parts database.

When users ask about:
- Part availability, stock, or inventory
- Searching for specific parts
- Part specifications or details

USE THE AVAILABLE TOOLS to get accurate, real-time information.

For general questions, answer directly without tools.

Be concise, accurate, and helpful."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    # Create agent
    agent = create_tool_calling_agent(
        llm=llm,
        tools=mcp_tools,
        prompt=prompt
    )
    
    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=mcp_tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3
    )
    
    return agent_executor, mcp_tools

# -----------------------------------------------------------------------------
# Process user query
# -----------------------------------------------------------------------------
async def process_query(query: str, agent_executor: AgentExecutor, chat_history: List) -> str:
    """Process user query through the agent"""
    try:
        # Convert chat history to LangChain message format
        lc_history = []
        for msg in chat_history[-6:]:  # Keep last 6 messages for context
            if msg["role"] == "user":
                lc_history.append(HumanMessage(content=msg["content"]))
            else:
                lc_history.append(AIMessage(content=msg["content"]))
        
        # Run agent
        result = await agent_executor.ainvoke({
            "input": query,
            "chat_history": lc_history
        })
        
        return result["output"]
        
    except Exception as e:
        logger.error(f"❌ Agent execution error: {e}")
        return f"Sorry, I encountered an error: {e}"

# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Japan Parts MCP Agent",
        page_icon="🇯🇵",
        layout="wide"
    )
    
    st.title("🇯🇵 Japan HQ Parts Assistant")
    st.caption("Powered by Groq Llama 3.3 + MCP")
    
    # Initialize agent (only once)
    if "agent_executor" not in st.session_state:
        with st.spinner("🔄 Initializing agent and connecting to MCP server..."):
            agent_executor, mcp_tools = asyncio.run(initialize_agent())
            st.session_state.agent_executor = agent_executor
            st.session_state.mcp_tools = mcp_tools
        
        # Show connection status
        if mcp_tools:
            st.success(f"✅ Connected! {len(mcp_tools)} MCP tool(s) available")
            with st.expander("📦 Available Tools"):
                for tool in mcp_tools:
                    st.write(f"**{tool.name}**: {tool.description}")
        else:
            st.warning("⚠️ No MCP tools available - agent will answer without tools")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about parts, stock, inventory..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response = asyncio.run(
                    process_query(
                        prompt,
                        st.session_state.agent_executor,
                        st.session_state.messages
                    )
                )
            st.markdown(response)
        
        # Add assistant message to history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Sidebar info
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        This agent can:
        - Search Japan HQ parts database
        - Check inventory and stock levels
        - Provide part specifications
        - Answer general questions
        """)
        
        st.divider()
        
        if st.button("🔄 Reset Chat"):
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        st.caption(f"Model: {GROQ_MODEL}")
        st.caption(f"MCP Server: {'✅ Connected' if st.session_state.mcp_tools else '❌ Disconnected'}")

if __name__ == "__main__":
    main()
