import asyncio
import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# --- NEW IMPORTS: These replace LLMChain ---
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

# --- CONFIGURATION ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
JAPAN_PARTS_SERVER_URL = os.environ.get("JAPAN_PARTS_SERVER_URL")
# Using the model from your Example 1
GROQ_MODEL = "llama-3.3-70b-versatile" 

# --- ASYNC SETUP (Connects to MCP) ---
async def create_agent_executor():
    """
    Connects to the MCP server, loads tools, and builds the Agent.
    """
    mcp_tools = []
    
    # 1. Load Tools from your MCP Server
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
        except Exception as e:
            print(f"MCP Connection Error: {e}")

    # 2. Initialize Groq (Same as your Example 2)
    llm = ChatGroq(
        model=GROQ_MODEL, 
        temperature=0, 
        api_key=GROQ_API_KEY
    )

    # 3. Define Prompt
    # We use 'agent_scratchpad' to let the AI "think" about tool results
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a helpful, friendly AI Agent. "
            "You have access to a Japan Parts Database. "
            "ALWAYS use the available tools to find prices, stock, or specs. "
            "Do not guess part details."
        )),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # 4. Create the Agent (The Brain)
    agent = create_tool_calling_agent(llm, mcp_tools, prompt)

    # 5. Create the Executor (The Runner)
    # This replaces 'conversation = LLMChain(...)' from your example
    agent_executor = AgentExecutor(agent=agent, tools=mcp_tools, verbose=True)
    
    return agent_executor, len(mcp_tools)

# --- HELPER: Run Async Agent in Sync Streamlit ---
async def run_chat(executor, user_input, history):
    result = await executor.ainvoke({
        "input": user_input, 
        "chat_history": history
    })
    return result["output"]

# --- MAIN UI (Based on your Example 2) ---
def main():
    st.set_page_config(page_title="Groq MCP Agent", layout="wide")
    st.title("😃 Groq + Japan Parts Agent")
    st.write("I'm your friendly chatbot! I can look up JDM parts for you. 🚀")

    # Sidebar just like your example
    st.sidebar.title('Customize')
    st.sidebar.write(f"**Model:** {GROQ_MODEL}")

    # Check for API Key
    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY not found in environment variables.")
        st.stop()

    # --- Session State Management ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Initialize Agent ONLY ONCE (to avoid reconnecting to MCP every click)
    if "agent_executor" not in st.session_state:
        with st.spinner("Connecting to MCP Server..."):
            executor, count = asyncio.run(create_agent_executor())
            st.session_state.agent_executor = executor
            if count > 0:
                st.sidebar.success(f"✅ Connected: {count} Tools")
            else:
                st.sidebar.warning("⚠️ No Tools Found (Check URL)")

    # --- Display Chat History ---
    for msg in st.session_state.messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

    # --- Handle User Input ---
    if prompt := st.chat_input("Ask about part prices..."):
        # 1. Show User Message
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Generate Response
        with st.chat_message("assistant"):
            with st.spinner("Checking database..."):
                # We exclude the *current* prompt from history passing to avoid duplication
                history = st.session_state.messages[:-1]
                
                response_text = asyncio.run(
                    run_chat(st.session_state.agent_executor, prompt, history)
                )
                
                st.markdown(response_text)
                st.session_state.messages.append(AIMessage(content=response_text))

if __name__ == "__main__":
    main()
