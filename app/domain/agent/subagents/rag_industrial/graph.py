from app.core.config import settings
from app.core.llm import LLMFactory
from langchain.agents import create_openai_tools_agent, AgentExecutor
from app.domain.agent.subagents.rag_industrial.tools import retrieve_documents
from app.domain.agent.subagents.rag_industrial.prompts import prompt
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

def create_industrial_graph():
    llm = LLMFactory.get_llm(role="subagent", temperature=0)
    
    tools = [retrieve_documents]
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)

    async def industrial_node(state):
        messages = state["messages"]
        # Convert LangGraph state messages to a string or list for the generic agent
        # Usually simplest to just pass the last user message or the whole history
        response = await agent_executor.ainvoke({"messages": messages})
        return {
            "messages": [AIMessage(content=response["output"], name="Industrial_RAG")]
        }

    return industrial_node
