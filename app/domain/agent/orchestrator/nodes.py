from app.core.config import settings
from app.core.llm import LLMFactory
from app.domain.agent.orchestrator.prompts import orchestrator_prompt
from app.domain.agent.orchestrator.tools import ask_knowledge_agent, ask_placeholder_agent
from langchain_core.utils.function_calling import convert_to_openai_tool

# List of tools available to the orchestrator
TOOLS = [ask_knowledge_agent, ask_placeholder_agent]
MEMBERS = "Knowledge_Base_Agent, Placeholder_Agent"

async def orchestrator_node(state, config, store):
    # Obtain session from config or dependency injection if possible
    # For now, we will use a workaround or ensure session is in config
    session = config.get("configurable", {}).get("session")
    llm = await LLMFactory.get_llm(role="orchestrator", temperature=0, session=session)
    user_id = config.get("configurable", {}).get("user_id", "default_user")

    # Bind tools to the LLM
    llm_with_tools = llm.bind_tools(TOOLS)
    
    # Chain: Prompt -> LLM with Tools
    chain = orchestrator_prompt | llm_with_tools
    
    response = await chain.ainvoke({
        "messages": state["messages"],
        "members": MEMBERS
    }, config=config)
    
    return {
        "messages": [response],
        "user_id": user_id
    }
