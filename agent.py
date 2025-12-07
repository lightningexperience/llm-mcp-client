# agent.py v2.0 - MCP-powered Groq Chat Agent (Heroku-ready)

import os
import sys
import json
import asyncio
import logging
import streamlit as st
from typing import List, Dict, Any

# Groq for LLM
from groq import Groq

# MCP Client
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
JAPAN_PARTS_SERVER_URL = os.environ.get("JAPAN_PARTS_SERVER_URL")
GROQ_MODEL = "llama-3.3-70b-versatile"

logger = logging.getLogger("MCPChatAgent")
logging.basicConfig(level=logging.INFO)

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is not set in Heroku Config Vars.")
    sys.exit(1)

# -----------------------------------------------------------------------------
# MCP Client Setup
# -----------------------------------------------------------------------------
class MCPToolManager:
    def __init__(self):
        self.session = None
        self.tools = []
        self.initialized = False
    
    async def initialize(self):
        """Initialize MCP connection and load tools"""
        if not JAPAN_PARTS_SERVER_URL:
            logger.warning("JAPAN_PARTS_SERVER_URL not set - MCP tools disabled")
            return
        
        try:
            # For streamable-http, we need a proxy
            # Using npx supergateway as the bridge
            server_params = StdioServerParameters(
                command="npx",
                args=[
                    "-y",
                    "supergateway",
                    "--streamableHttp",
                    JAPAN_PARTS_SERVER_URL
                ]
            )
            
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    self.session = session
                    await session.initialize()
                    
                    # List available tools
                    tools_result = await session.list_tools()
                    self.tools = tools_result.tools
                    self.initialized = True
                    
                    logger.info(f"MCP: Loaded {len(self.tools)} tool(s)")
                    for tool in self.tools:
                        logger.info(f"  - {tool.name}: {tool.description}")
                        
        except Exception as e:
            logger.error(f"MCP initialization failed: {e}")
            self.initialized = False
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Call an MCP tool"""
        if not self.initialized:
            return "MCP tools not available"
        
        try:
            result = await self.session.call_tool(tool_name, arguments)
            # Extract text from result
            if result.content:
                return str(result.content[0].text)
            return "No result from tool"
        except Exception as e:
            logger.error(f"Tool call error: {e}")
            return f"Error calling tool: {e}"
    
    def get_tools_description(self) -> str:
        """Get formatted description of available tools"""
        if not self.tools:
            return "No MCP tools available."
        
        descriptions = []
        for tool in self.tools:
            params = json.dumps(tool.inputSchema.get("properties", {}), indent=2)
            descriptions.append(
                f"Tool: {tool.name}\n"
                f"Description: {tool.description}\n"
                f"Parameters: {params}\n"
            )
        return "\n".join(descriptions)

# -----------------------------------------------------------------------------
# Groq LLM Integration
# -----------------------------------------------------------------------------
def call_groq_with_tools(user_message: str, mcp_manager: MCPToolManager) -> str:
    """Call Groq LLM with tool awareness"""
    
    client = Groq(api_key=GROQ_API_KEY)
    
    # Build system prompt with tool info
    tools_info = mcp_manager.get_tools_description()
    system_prompt = f"""You are a helpful assistant for Japan HQ auto parts inventory.

You have access to the following tools:
{tools_info}

When the user asks about parts, inventory, stock, or searches:
1. Identify if you need to use a tool
2. If yes, respond EXACTLY in this format:
USE_TOOL: tool_name
ARGUMENTS: {{"param": "value"}}

3. If no tool needed, answer directly.

Be concise and helpful."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=1024
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        logger.error(f"Groq API error: {e}")
        return f"Error contacting LLM: {e}"

async def process_user_query(user_message: str, mcp_manager: MCPToolManager) -> str:
    """Process user query with tool calling if needed"""
    
    # Get LLM response
    llm_response = call_groq_with_tools(user_message, mcp_manager)
    
    # Check if LLM wants to use a tool
    if "USE_TOOL:" in llm_response:
        try:
            lines = llm_response.split("\n")
            tool_name = None
            arguments = {}
            
            for line in lines:
                if line.startswith("USE_TOOL:"):
                    tool_name = line.replace("USE_TOOL:", "").strip()
                elif line.startswith("ARGUMENTS:"):
                    args_str = line.replace("ARGUMENTS:", "").strip()
                    arguments = json.loads(args_str)
            
            if tool_name:
                # Call the tool
                tool_result = await mcp_manager.call_tool(tool_name, arguments)
                
                # Get final response from LLM with tool result
                final_prompt = f"""The user asked: {user_message}

You called the tool '{tool_name}' and got this result:
{tool_result}

Now provide a helpful answer to the user based on this result."""

                client = Groq(api_key=GROQ_API_KEY)
                final_response = client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[{"role": "user", "content": final_prompt}],
                    temperature=0.3
                )
                
                return final_response.choices[0].message.content.strip()
        
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return f"I tried to use a tool but encountered an error: {e}"
    
    # No tool needed, return LLM response directly
    return llm_response

# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Japan Parts MCP Agent", layout="wide")
    st.title("🇯🇵 Japan HQ Parts Assistant (Groq + MCP)")
    
    # Initialize MCP manager
    if "mcp_manager" not in st.session_state:
        st.session_state.mcp_manager = MCPToolManager()
        asyncio.run(st.session_state.mcp_manager.initialize())
        
        if st.session_state.mcp_manager.initialized:
            st.success(f"✅ Connected to MCP server with {len(st.session_state.mcp_manager.tools)} tool(s)")
        else:
            st.warning("⚠️ MCP tools not available - LLM will answer without tools")
    
    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
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
        with st.spinner("Thinking..."):
            response = asyncio.run(
                process_user_query(prompt, st.session_state.mcp_manager)
            )
        
        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

if __name__ == "__main__":
    main()
