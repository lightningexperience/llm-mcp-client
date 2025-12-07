# agent.py - MCP Chat Agent with Groq (using proven groq package)

import os
import sys
import asyncio
import json
import logging
import streamlit as st
from typing import List, Dict, Any

# Use standalone Groq (like your working A2A server)
from groq import Groq

# LangChain MCP
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
# MCP Tool Manager
# -----------------------------------------------------------------------------
class MCPToolManager:
    def __init__(self):
        self.tools = []
        self.initialized = False
    
    async def initialize(self):
        """Load MCP tools"""
        if not JAPAN_PARTS_SERVER_URL:
            logger.warning("JAPAN_PARTS_SERVER_URL not set - MCP disabled")
            return
        
        try:
            client = MultiServerMCPClient(
                servers={
                    "japan_parts_db": {
                        "url": JAPAN_PARTS_SERVER_URL,
                        "transport": "streamable-http"
                    }
                }
            )
            
            self.tools = await load_mcp_tools(client)
            self.initialized = True
            logger.info(f"✅ Loaded {len(self.tools)} MCP tool(s)")
            
        except Exception as e:
            logger.error(f"❌ MCP error: {e}")
    
    def get_tools_for_llm(self) -> List[Dict]:
        """Format tools for Groq function calling"""
        groq_tools = []
        for tool in self.tools:
            groq_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.args_schema.schema() if hasattr(tool, 'args_schema') else {}
                }
            })
        return groq_tools
    
    async def call_tool(self, tool_name: str, arguments: Dict) -> str:
        """Execute MCP tool"""
        try:
            for tool in self.tools:
                if tool.name == tool_name:
                    result = await tool.ainvoke(arguments)
                    return str(result)
            return f"Tool {tool_name} not found"
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return f"Error: {e}"

# -----------------------------------------------------------------------------
# Groq Chat with Tool Calling
# -----------------------------------------------------------------------------
async def chat_with_tools(
    user_message: str,
    mcp_manager: MCPToolManager,
    chat_history: List[Dict]
) -> str:
    """Process user message with Groq + MCP tools"""
    
    client = Groq(api_key=GROQ_API_KEY)
    
    # Build messages
    messages = [
        {
            "role": "system",
            "content": """You are a helpful assistant for Japan HQ auto parts inventory.

When users ask about parts, inventory, stock, or search:
- Use the available tools to get accurate information
- Provide clear, concise answers based on tool results

For general questions, answer directly."""
        }
    ]
    
    # Add chat history (last 6 messages)
    for msg in chat_history[-6:]:
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    
    # Add current user message
    messages.append({
        "role": "user",
        "content": user_message
    })
    
    # Get tools
    tools = mcp_manager.get_tools_for_llm() if mcp_manager.initialized else None
    
    # First LLM call
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto" if tools else None,
            temperature=0.3
        )
        
        response_message = response.choices[0].message
        
        # Check if tool call needed
        if response_message.tool_calls:
            # Execute tool calls
            messages.append(response_message)
            
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                logger.info(f"Calling tool: {function_name} with args: {function_args}")
                
                # Execute tool
                tool_result = await mcp_manager.call_tool(function_name, function_args)
                
                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": tool_result
                })
            
            # Second LLM call with tool results
            final_response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0.3
            )
            
            return final_response.choices[0].message.content
        
        # No tool call needed
        return response_message.content
        
    except Exception as e:
        logger.error(f"Groq API error: {e}")
        return f"Sorry, I encountered an error: {e}"

# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Japan Parts Agent",
        page_icon="🇯🇵",
        layout="wide"
    )
    
    st.title("🇯🇵 Japan HQ Parts Assistant")
    st.caption(f"Powered by Groq {GROQ_MODEL}")
    
    # Initialize MCP manager
    if "mcp_manager" not in st.session_state:
        with st.spinner("🔄 Connecting to MCP server..."):
            manager = MCPToolManager()
            asyncio.run(manager.initialize())
            st.session_state.mcp_manager = manager
        
        if manager.initialized:
            st.success(f"✅ MCP Ready - {len(manager.tools)} tool(s) available")
        else:
            st.warning("⚠️ MCP unavailable - using LLM only")
    
    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # User input
    if prompt := st.chat_input("Ask about parts, stock, or inventory..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response = asyncio.run(
                    chat_with_tools(
                        prompt,
                        st.session_state.mcp_manager,
                        st.session_state.messages
                    )
                )
            st.markdown(response)
        
        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ Info")
        st.write(f"""
        **Model:** {GROQ_MODEL}
        
        **MCP Status:** {'✅ Connected' if st.session_state.mcp_manager.initialized else '❌ Disconnected'}
        
        **Tools Available:** {len(st.session_state.mcp_manager.tools) if st.session_state.mcp_manager.initialized else 0}
        """)
        
        if st.button("🔄 Clear Chat"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()
