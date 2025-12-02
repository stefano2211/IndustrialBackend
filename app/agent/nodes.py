from app.agent.state import AgentState
from app.agent.tools import retrieve_documents, use_custom_tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode
from app.config import settings
from app.core.db import get_session
from app.services.tool_config_service import ToolConfigService

# Initialize LLM with tools
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=settings.openrouter_api_key,
    model=settings.openrouter_model,
    temperature=0
) 

from loguru import logger
logger.info(f"LLM Initialized with base_url: {llm.openai_api_base}, model: {llm.model_name}")

tools = [retrieve_documents, use_custom_tool]
llm_with_tools = llm.bind_tools(tools)

async def agent(state: AgentState):
    """
    Invokes the model with the current state (messages).
    """
    messages = state["messages"]
    
    # Fetch available tools dynamically
    available_tools_desc = ""
    async for session in get_session():
        service = ToolConfigService(session)
        tools_list = await service.get_all()
        if tools_list:
            available_tools_desc = "\nAvailable External Tools (use `use_custom_tool` with the name):\n"
            for t in tools_list:
                available_tools_desc += f"- {t.name}: {t.description}\n"
        break # Only need one session

    # System prompt for the main orchestrator
    system_prompt = SystemMessage(content=f"""You are a financial analysis orchestrator. You have access to these tools:
1. `retrieve_documents`: Use this to find information in internal financial documents (PDFs, reports).
2. `use_custom_tool`: Use this to fetch data from external APIs.

{available_tools_desc}

Decide which tool to use based on the user's query:
- If the user asks for current market data or external info, check if a suitable tool is available in the list above and use `use_custom_tool` with its name.
- If the user asks about specific internal reports, strategic plans, or document content, use `retrieve_documents`.

You are responsible for analyzing the data returned by these tools to answer the user's question comprehensively.""")
    
    response = await llm_with_tools.ainvoke([system_prompt] + messages)
    return {"messages": [response]}

# ToolNode handles the execution of tools
tool_node = ToolNode(tools)
