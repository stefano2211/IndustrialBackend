from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from app.config import settings
from app.domain.agent.subagents.rag_financial.tools import retrieve_documents
from app.domain.agent.subagents.rag_financial.prompts import prompt
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

def create_financial_graph():
    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.openrouter_api_key,
        model=settings.openrouter_model,
        temperature=0
    )
    
    tools = [retrieve_documents]
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)

    async def financial_node(state):
        messages = state["messages"]
        # Convert LangGraph state messages to a string or list for the generic agent
        # Usually simplest to just pass the last user message or the whole history
        response = await agent_executor.ainvoke({"messages": messages})
        return {
            "messages": [AIMessage(content=response["output"], name="Financial_RAG")]
        }

    return financial_node
