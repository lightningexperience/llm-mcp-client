import asyncio
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

# --- 1. CONFIGURATION: Use Environment Variables ---
# Recommended way to provide secrets and URLs (best for Heroku)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
JAPAN_PARTS_SERVER_URL = os.environ.get("JAPAN_PARTS_SERVER_URL")

# --- Error Check ---
if not OPENAI_API_KEY or not JAPAN_PARTS_SERVER_URL:
    raise ValueError(
        "Please set OPENAI_API_KEY and JAPAN_PARTS_SERVER_URL "
        "environment variables in your Heroku settings."
    )
# --- End Config ---

async def run_agent_chat(user_query: str):
    """
    Initializes the MCP client, loads tools, and runs the LLM agent.
    """
    
    # 2. Initialize the MultiServerMCPClient
    # This is the MCP Client that connects to your remote server.
    print(f"Connecting to MCP server at: {JAPAN_PARTS_SERVER_URL}")
    
    mcp_client = MultiServerMCPClient(
        servers={
            "japan_parts": { # Give your server a unique name (label)
                "url": JAPAN_PARTS_SERVER_URL,
                "transport": "streamable-http"
            }
        }
    )

    # 3. Load MCP Tools and Translate for the LLM
    # The adapter connects, discovers all tools on the server, and converts their
    # definitions into the JSON Schema format that the LLM (like GPT-4o) expects.
    try:
        mcp_tools = await load_mcp_tools(mcp_client)
    except Exception as e:
        print(f"Failed to load tools from MCP server: {e}")
        return "Sorry, the Japan Parts database is currently unavailable."

    if not mcp_tools:
        return "Error: No tools were found on the MCP server. Cannot search for parts."

    print(f"✅ Successfully loaded {len(mcp_tools)} tool(s) from the MCP server.")

    # 4. Configure the LLM and Agent Prompt
    llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=OPENAI_API_KEY)
    
    # CRITICAL: The system prompt tells the LLM WHEN and WHY to use the tool.
    system_prompt = (
        "You are an expert parts agent. Your primary role is to answer questions about "
        "Japan parts pricing, stock, or specifications. "
        "ALWAYS use the loaded tool(s) for these queries. "
        "Do not guess or use your general knowledge for parts data."
    )
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    # 5. Create the Agent
    # The agent uses the LLM's function-calling capability and the loaded tools.
    agent = create_react_agent(llm, mcp_tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=mcp_tools, verbose=True)

    # 6. Run the Agent (Handles the Full Tool-Calling Loop)
    print("\n--- Running Agent ---")
    result = await agent_executor.ainvoke({"input": user_query})

    return result['output']

# --- Main Execution ---
if __name__ == "__main__":
    # Example Query - The agent will decide to call the MCP tool here
    query = "What is the price and current stock count for the JDM-500 turbo manifold?"
    
    # In your UI, you would call this function when the user submits a question.
    final_answer = asyncio.run(run_agent_chat(query))
    
    print("\n--- Final Answer ---")
    print(final_answer)
