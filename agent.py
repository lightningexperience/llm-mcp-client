import asyncio
import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Modern Agent Imports (Requires langchain >= 0.2.14)
from langchain.agents import create_tool_calling_agent, AgentExecutor

# MCP Imports
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

# --- Configuration ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
JAPAN_PARTS_SERVER_URL = os.environ.get("JAPAN_PARTS_SERVER_URL")
GROQ_MODEL = "llama-3.3-70b-versatile"

# --- Async Setup for MCP ---
async def build_agent_executor():
    """
    Connects to MCP, loads tools, and creates the LangChain Agent.
    Returns: The AgentExecutor (runnable object).
    """
    mcp_tools = []
    
    # 1. Load MCP Tools (if URL is set)
    if JAPAN_PARTS_SERVER_URL:
        try:
            client = MultiServerMCPClient(
                servers={
                    "japan_parts": {
                        "url": JAPAN_PARTS_SERVER_URL,
                        "transport": "streamable-http"
                    }
                }
            )
            mcp_tools = await load_mcp_tools(client)
            print(f"DEBUG: Loaded {len(mcp_tools)} tools from MCP.")
        except Exception as e:
            print(f"DEBUG: MCP Connection failed: {e}")

    # 2. Setup LLM
    llm = ChatGroq(
        model=GROQ_MODEL, 
        temperature=0, 
        api_key=GROQ_API_KEY
    )

    # 3. Setup Prompt (Must include 'chat_history' and 'agent_scratchpad')
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a friendly, helpful AI Agent. "
            "You have access to a Japan Parts Database via tools. "
            "ALWAYS use the tool if the user asks about parts, prices, or stock. "
            "If the tool returns data, answer based on that data."
        )),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # 4. Create Agent
    agent = create_tool_calling_agent(llm, mcp_tools, prompt)

    # 5. Create Executor
    agent_executor = AgentExecutor(agent=agent, tools=mcp_tools, verbose=True)
    
    return agent_executor, len(mcp_tools)

# --- Helper to Run Async in Streamlit ---
async def run_interaction(agent_executor, user_input, chat_history):
    response = await agent_executor.ainvoke({
        "input": user_input,
        "chat_history": chat_history
    })
    return response["output"]

# --- Main Streamlit UI ---
def main():
    st.set_page_config(page_title="Friendly MCP Agent", layout="wide")
    st.title("Hey there! 🇯🇵")
    st.write("I'm your friendly AI agent with access to the Japan Parts Database! 🚀")

    # Check API Keys
    if not GROQ_API_KEY:
        st.error("Error: GROQ_API_KEY not found in environment.")
        st.stop()

    # --- Initialize Session State ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Initialize the Agent ONLY ONCE
    if "agent_executor" not in st.session_state:
        with st.spinner("Connecting to MCP Server..."):
            # We run the async setup synchronously here once
            executor, tool_count = asyncio.run(build_agent_executor())
            st.session_state.agent_executor = executor
            if tool_count > 0:
                st.success(f"Connected! {tool_count} Tools Loaded.")
            else:
                st.warning("No tools loaded (Check Server URL). Running in chat-only mode.")

    # --- Display Chat History ---
    for msg in st.session_state.messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

    # --- Handle User Input ---
    if prompt := st.chat_input("Ask about parts, or just say hi..."):
        # 1. Display User Message
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Run Agent
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Run the async agent logic
                    response_text = asyncio.run(
                        run_interaction(
                            st.session_state.agent_executor, 
                            prompt, 
                            st.session_state.messages[:-1] # Pass history excluding current prompt
                        )
                    )
                    st.markdown(response_text)
                    
                    # 3. Save Assistant Response
                    st.session_state.messages.append(AIMessage(content=response_text))
                except Exception as e:
                    st.error(f"Something went wrong: {e}")

if __name__ == "__main__":
    main()
