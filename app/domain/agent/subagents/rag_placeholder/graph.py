from app.core.config import settings
from app.core.llm import LLMFactory
from app.domain.agent.subagents.rag_placeholder.prompts import prompt
from langchain_core.messages import AIMessage

async def placeholder_node(state):
    llm = LLMFactory.get_llm(role="subagent", temperature=0)
    
    # Simple chain: Prompt -> LLM
    chain = prompt | llm
    response = await chain.ainvoke({"messages": state["messages"]})
    
    return {
        "messages": [AIMessage(content=response.content, name="Placeholder_RAG")]
    }
