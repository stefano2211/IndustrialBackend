from typing import Annotated, Any
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.api import deps
from app.persistence.db import get_session
from app.domain.schemas.user import UserRead, UserUpdate
from app.domain.schemas.user import User
from app.domain.proactiva.services.user_service import UserService

router = APIRouter()

@router.get("/me", response_model=UserRead)
async def read_user_me(
    current_user: Annotated[User, Depends(deps.get_current_user)]
) -> Any:
    """
    Get current user.
    """
    return current_user

@router.put("/me", response_model=UserRead)
async def update_user_me(
    *,
    session: Annotated[AsyncSession, Depends(get_session)],
    user_in: UserUpdate,
    current_user: Annotated[User, Depends(deps.get_current_user)],
) -> Any:
    """
    Update own user.
    """
    user_service = UserService(session)
    user = await user_service.update_user(user=current_user, user_in=user_in)
    return user
