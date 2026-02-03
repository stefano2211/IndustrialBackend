from typing import Annotated, Any
from fastapi import APIRouter, Depends
from sqlmodel import Session
from app.api import deps
from app.core.database import get_session
from app.domain.schemas.user import UserRead, UserUpdate
from app.domain.models.user import User
from app.domain.services.user_service import UserService

router = APIRouter()

@router.get("/me", response_model=UserRead)
def read_user_me(
    current_user: Annotated[User, Depends(deps.get_current_user)]
) -> Any:
    """
    Get current user.
    """
    return current_user

@router.put("/me", response_model=UserRead)
def update_user_me(
    *,
    session: Annotated[Session, Depends(get_session)],
    user_in: UserUpdate,
    current_user: Annotated[User, Depends(deps.get_current_user)],
) -> Any:
    """
    Update own user.
    """
    user_service = UserService(session)
    user = user_service.update_user(user=current_user, user_in=user_in)
    return user
