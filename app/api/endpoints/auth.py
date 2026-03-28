from typing import Annotated, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.api import deps
from app.persistence.db import get_session
from app.core.security import create_access_token
from app.domain.schemas.token import Token
from app.domain.schemas.user import UserCreate, UserRead, UserLogin
from app.domain.services.user_service import UserService
from app.persistence.repositories.settings_repository import SettingsRepository

router = APIRouter()

@router.post("/login", response_model=Token)
async def login(
    session: Annotated[AsyncSession, Depends(get_session)],
    login_data: UserLogin
) -> Any:
    """
    JSON body login, get an access token for future requests.
    """
    user_service = UserService(session)
    user = await user_service.authenticate(
        email=login_data.email, password=login_data.password
    )
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect email or password")
    elif not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    
    return {
        "access_token": create_access_token(user.email),
        "token_type": "bearer",
    }

@router.post("/register", response_model=UserRead)
async def register_user(
    *,
    session: Annotated[AsyncSession, Depends(get_session)],
    user_in: UserCreate,
) -> Any:
    """
    Create new user.
    First registered user automatically becomes admin.
    """
    settings_repo = SettingsRepository(session)
    system_settings = await settings_repo.get_settings()
    
    if not system_settings.auth_enable_sign_ups:
        raise HTTPException(
            status_code=403,
            detail="Public Sign-ups are currently disabled by the Administrator.",
        )

    user_service = UserService(session)
    user = await user_service.get_by_email(email=user_in.email)
    if user:
        raise HTTPException(
            status_code=400,
            detail="The user with this email already exists in the system",
        )

    # First user becomes admin automatically
    existing_users = await user_service.list_users()
    if len(existing_users) == 0:
        user_in.is_superuser = True

    try:
        user = await user_service.create_user(user_in=user_in)
        return user
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
