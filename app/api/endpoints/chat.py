"""Chat endpoint — thin handler delegating to AgentService + ConversationService."""

import json
import uuid
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.schemas.api import ChatRequest, ChatResponse
from app.domain.schemas.conversation import ChatMessage
from app.domain.services.agent_service import AgentService
from app.domain.services.conversation_service import ConversationService
from app.api import deps
from app.persistence.db import get_session
from app.domain.schemas.user import User

router = APIRouter()

_agent_service = AgentService()


async def get_conversation_service(
    session: AsyncSession = Depends(get_session),
) -> ConversationService:
    return ConversationService(session)


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    fastapi_request: Request,
    current_user: User = Depends(deps.get_current_user),
    session: AsyncSession = Depends(get_session),
    conv_service: ConversationService = Depends(get_conversation_service),
):
    """Chat with the AI Agent (non-streaming, full response)."""
    try:
        user_id = str(current_user.id)
        thread_id = request.thread_id or str(uuid.uuid4())

        await conv_service.get_or_create_conversation(
            thread_id=thread_id,
            user_id=current_user.id,
            title=request.query[:80] if request.query else "New Chat",
        )

        session.add(ChatMessage(thread_id=thread_id, role="user", content=request.query, model_id=request.model_id))
        await session.commit()

        checkpointer = getattr(fastapi_request.app.state, "checkpointer", None)
        store = getattr(fastapi_request.app.state, "store", None)

        answer, resolved_model_id = await _agent_service.invoke(
            user_id=user_id,
            thread_id=thread_id,
            query=request.query,
            knowledge_base_id=request.knowledge_base_id,
            mcp_source_id=request.mcp_source_id,
            session=session,
            checkpointer=checkpointer,
            store=store,
            params=request.params,
            model_id=request.model_id,
        )

        session.add(ChatMessage(thread_id=thread_id, role="assistant", content=answer, model_id=resolved_model_id))
        await session.commit()

        return ChatResponse(answer=answer, sources=[], thread_id=thread_id)

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
async def chat_stream_endpoint(
    request: ChatRequest,
    fastapi_request: Request,
    current_user: User = Depends(deps.get_current_user),
    session: AsyncSession = Depends(get_session),
    conv_service: ConversationService = Depends(get_conversation_service),
):
    """
    Chat with the AI Agent — Server-Sent Events (SSE) streaming.
    
    Sends events:
      - data: {"type": "meta", "thread_id": "..."}
      - data: {"type": "token", "content": "..."}
      - data: {"type": "done", "full_content": "..."}
      - data: {"type": "error", "detail": "..."}
    """
    user_id = str(current_user.id)
    thread_id = request.thread_id or str(uuid.uuid4())

    try:
        await conv_service.get_or_create_conversation(
            thread_id=thread_id,
            user_id=current_user.id,
            title=request.query[:80] if request.query else "New Chat",
        )

        session.add(ChatMessage(thread_id=thread_id, role="user", content=request.query, model_id=request.model_id))
        await session.commit()
    except Exception as e:
        logger.error(f"Error setting up stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    checkpointer = getattr(fastapi_request.app.state, "checkpointer", None)
    store = getattr(fastapi_request.app.state, "store", None)

    async def event_generator():
        full_content = ""
        try:
            yield f"data: {json.dumps({'type': 'meta', 'thread_id': thread_id})}\n\n"

            resolved_model_id = "default"
            async for chunk in _agent_service.stream(
                user_id=user_id,
                thread_id=thread_id,
                query=request.query,
                knowledge_base_id=request.knowledge_base_id,
                mcp_source_id=request.mcp_source_id,
                session=session,
                checkpointer=checkpointer,
                store=store,
                params=request.params,
                model_id=request.model_id,
            ):
                if isinstance(chunk, dict) and "model_id" in chunk:
                    resolved_model_id = chunk["model_id"]
                    continue
                    
                full_content += chunk
                yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"

            session.add(ChatMessage(thread_id=thread_id, role="assistant", content=full_content, model_id=resolved_model_id))
            await session.commit()

            yield f"data: {json.dumps({'type': 'done', 'full_content': full_content})}\n\n"

        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'detail': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
