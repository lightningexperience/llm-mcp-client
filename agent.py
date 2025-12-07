import asyncio
import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# These imports will work once runtime.txt forces a rebuild
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

# --- CONFIGURATION ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
JAPAN_PARTS_SERVER_URL = os.environ.get("JAPAN_PARTS_SERVER_URL")
GROQ_MODEL = "llama-3.3-70b-versatile"

# --- ASYNC AGENT SETUP ---
async def create_agent_executor():
    # 1. Load MCP Tools
    mcp_tools = []
    if JAPAN_PARTS_SERVER_URL:
        try:
            client = MultiServerMCPClient(
                servers={"japan_parts": {"url": JAPAN_PARTS_SERVER_URL, "transport": "streamable-http"}}
            )
            mcp_tools = await load_mcp_tools(client)
        except Exception as e:
            print(f"MCP Error: {e}")

    # 2. Setup LLM
    llm = ChatGroq(model=GROQ_MODEL, temperature=0, api_key=GROQ_API_KEY)

    # 3. Setup Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant with access to a Japan Parts Database. Use tools for any parts questions."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # 4. Create Agent & Executor
    agent = create_tool_calling_agent(llm, mcp_tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=mcp_tools, verbose=True)
    
    return agent_executor

# --- HELPER: Run Async in Streamlit ---
async def run_chat(executor, user_input, history):
    return await executor.ainvoke({"input": user_input, "chat_history": history})

# --- UI ---
def main():
    st.set_page_config(page_title="Groq MCP Agent")
    st.title("🇯🇵 Groq + MCP Parts Agent")

    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY missing.")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Init Agent
    if "agent_executor" not in st.session_state:
        with st.spinner("Connecting to MCP..."):
            st.session_state.agent_executor = asyncio.run(create_agent_executor())

    # Chat Loop
    for msg in st.session_state.messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

    if prompt := st.chat_input("Ask about parts..."):
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = asyncio.run(run_chat(
                    st.session_state.agent_executor, 
                    prompt, 
                    st.session_state.messages[:-1]
                ))
                st.markdown(response["output"])
                st.session_state.messages.append(AIMessage(content=response["output"]))

if __name__ == "__main__":
    main()
