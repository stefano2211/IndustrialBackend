from fastapi import APIRouter, HTTPException, Depends
from app.domain.schemas.api import ChatRequest, ChatResponse
from app.domain.agent.workflow import app as agent_app
from langchain_core.messages import HumanMessage
from loguru import logger
from app.api import deps
from app.domain.schemas.user import User

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
        # User identification
        user_id = request.user_id or str(current_user.id)
        thread_id = request.thread_id or "default_thread"

        # Convert user input to LangChain message
        messages = [HumanMessage(content=request.query)]
        
        # Config for LangGraph (Threading + Store)
        config = {
            "configurable": {
                "thread_id": thread_id,
                "user_id": user_id
            }
        }

        # Invoke the agent graph with persistence config
        response = await agent_app.ainvoke({"messages": messages}, config=config)
        
        # Extract the final response content
        final_message = response["messages"][-1].content
        
        return ChatResponse(
            answer=final_message, 
            sources=[],
            thread_id=thread_id # Need to add this to ChatResponse too or just return it
        )

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
