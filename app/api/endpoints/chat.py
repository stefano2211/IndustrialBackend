import uuid
from fastapi import APIRouter, HTTPException, Depends
from sqlmodel import Session, select
from app.domain.schemas.api import ChatRequest, ChatResponse
from app.domain.schemas.conversation import Conversation, ChatMessage
from app.domain.agent.workflow import app as agent_app
from langchain_core.messages import HumanMessage
from loguru import logger
from app.api import deps
from app.core.database import get_session
from app.domain.schemas.user import User
from typing import Annotated
from datetime import datetime

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    current_user: User = Depends(deps.get_current_user),
    session: Session = Depends(get_session),
):
    """
    Chat with the AI Agent (Orchestrator).
    Auto-creates a Conversation record and persists messages to DB.
    """
    try:
        user_id = str(current_user.id)
        thread_id = request.thread_id or str(uuid.uuid4())

        # Auto-create Conversation if it doesn't exist
        statement = select(Conversation).where(Conversation.thread_id == thread_id)
        conversation = session.exec(statement).first()
        if not conversation:
            conversation = Conversation(
                user_id=current_user.id,
                thread_id=thread_id,
                title=request.query[:80] if request.query else "New Chat",
            )
            session.add(conversation)
            session.commit()
            session.refresh(conversation)

        # Update timestamp on every message
        conversation.updated_at = datetime.utcnow()
        session.add(conversation)
        session.commit()

        # Save user message to DB
        user_msg = ChatMessage(thread_id=thread_id, role="user", content=request.query)
        session.add(user_msg)
        session.commit()

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
        
        # Save assistant message to DB
        assistant_msg = ChatMessage(thread_id=thread_id, role="assistant", content=final_message)
        session.add(assistant_msg)
        session.commit()
        
        return ChatResponse(
            answer=final_message, 
            sources=[],
            thread_id=thread_id
        )

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

