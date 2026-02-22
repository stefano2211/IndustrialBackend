from langchain_openai import ChatOpenAI
from app.core.config import settings
from app.domain.agent.orchestrator.prompts import orchestrator_prompt
from app.domain.agent.orchestrator.tools import ask_industrial_agent, ask_placeholder_agent
from langchain_core.utils.function_calling import convert_to_openai_tool

# List of tools available to the orchestrator
TOOLS = [ask_industrial_agent, ask_placeholder_agent]
MEMBERS = "Industrial_RAG, Placeholder_RAG"

def orchestrator_node(state):
    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.openrouter_api_key,
        model=settings.openrouter_model,
        temperature=0
    )

    # Bind tools to the LLM
    llm_with_tools = llm.bind_tools(TOOLS)
    
    # Chain: Prompt -> LLM with Tools
    chain = orchestrator_prompt | llm_with_tools
    
    response = chain.invoke({
        "messages": state["messages"],
        "members": MEMBERS
    })
    
    return {"messages": [response]}
