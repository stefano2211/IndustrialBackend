from fastapi import APIRouter, HTTPException, Depends
from app.domain.schemas.api import ChatRequest, ChatResponse
from app.domain.agent.workflow import app as agent_app
from langchain_core.messages import HumanMessage
from loguru import logger
from app.api import deps
from app.domain.models.user import User

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    current_user: User = Depends(deps.get_current_user)
):
    """
    Chat with the AI Agent (Orchestrator).
    The agent will route your request to the appropriate sub-agent (Financial RAG or Placeholder).
    """
    try:
        # Convert user input to LangChain message
        messages = [HumanMessage(content=request.query)]
        
        # Invoke the agent graph
        response = await agent_app.ainvoke({"messages": messages})
        
        # Extract the final response content
        # The response is typically the last message in the list
        final_message = response["messages"][-1].content
        
        # For now, sources are empty as we haven't extracted them from the graph state yet
        return ChatResponse(answer=final_message, sources=[])

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
