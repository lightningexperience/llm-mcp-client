import asyncio
import os
import sys
from langchain_groq import ChatGroq # NEW: Import Groq's LangChain integration
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_agent
from langchain.agents import AgentExecutor # Try this first, as it's the intended location
# If the error persists, try importing AgentExecutor from the new consolidated executor module
# from langchain.agents import AgentExecutor
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

# --- 1. CONFIGURATION: Use Heroku Environment Variables ---
# The environment variables must be set in your Heroku Config Vars.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
JAPAN_PARTS_SERVER_URL = os.environ.get("JAPAN_PARTS_SERVER_URL")

# The specific model you requested
GROQ_MODEL = "llama-3.3-70b-versatile" 

# --- Error Check ---
if not GROQ_API_KEY:
    print("Error: GROQ_API_KEY environment variable not set. Cannot run LLM.")
    sys.exit(1)
if not JAPAN_PARTS_SERVER_URL:
    print("Error: JAPAN_PARTS_SERVER_URL environment variable not set. Cannot run MCP client.")
    # Allow running for general queries, but print a warning
    # sys.exit(1) # Uncomment this if you require the parts database to be available
# --- End Config ---


async def run_agent_chat(user_query: str):
    """
    Initializes the Groq LLM and the MCP client, then runs the agent.
    """
    
    # 2. Initialize the MultiServerMCPClient
    print(f"Connecting to MCP server at: {JAPAN_PARTS_SERVER_URL}")
    mcp_client = MultiServerMCPClient(
        servers={
            "japan_parts_db": { # A label for the server
                "url": JAPAN_PARTS_SERVER_URL,
                "transport": "streamable-http"
            }
        }
    )

    # 3. Load MCP Tools and Translate for the LLM
    try:
        mcp_tools = await load_mcp_tools(mcp_client)
    except Exception as e:
        print(f"⚠️ Warning: Failed to load tools from MCP server. Will only answer with general knowledge. Error: {e}")
        mcp_tools = []

    if mcp_tools:
        print(f"✅ Successfully loaded {len(mcp_tools)} tool(s) from the MCP server.")
    
    # 4. Configure the LLM (Using ChatGroq)
    # LangChain handles passing the API key from the environment variable automatically.
    llm = ChatGroq(model=GROQ_MODEL, temperature=0) # Groq is very fast, low temperature is ideal for agents.
    
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

    # 5. Create the Agent
    # The AgentExecutor handles the multi-step process: LLM -> Tool Call -> MCP Execution -> LLM Synthesis.
    agent = create_react_agent(llm, mcp_tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=mcp_tools, verbose=True)

    # 6. Run the Agent
    print(f"\n--- Running Agent for Query: '{user_query}' ---")
    result = await agent_executor.ainvoke({"input": user_query})

    return result['output']

# --- Main Execution Loop (Simulating a UI Interaction) ---
if __name__ == "__main__":
    
    # Example Query - This should trigger the MCP tool call
    test_query = "What is the price and stock count for the JDM-500 turbo manifold?"
    
    # Run the main function
    final_answer = asyncio.run(run_agent_chat(test_query))
    
    print("\n--- Final Agent Answer ---")
    print(final_answer)
