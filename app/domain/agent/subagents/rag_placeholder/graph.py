from langchain_openai import ChatOpenAI
from app.config import settings
from app.domain.agent.subagents.rag_placeholder.prompts import prompt
from langchain_core.messages import AIMessage

async def placeholder_node(state):
    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.openrouter_api_key,
        model=settings.openrouter_model,
        temperature=0
    )
    
    # Simple chain: Prompt -> LLM
    chain = prompt | llm
    response = await chain.ainvoke({"messages": state["messages"]})
    
    return {
        "messages": [AIMessage(content=response.content, name="Placeholder_RAG")]
    }
