from typing import Annotated, Any
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlmodel import Session
from app.api import deps
from app.core.database import get_session
from app.core.security import create_access_token
from app.domain.schemas.token import Token
from app.domain.schemas.user import UserCreate, UserRead, UserLogin
from app.domain.services.user_service import UserService

router = APIRouter()

@router.post("/login", response_model=Token)
def login_access_token(
    session: Annotated[Session, Depends(get_session)],
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()]
) -> Any:
    """
    OAuth2 compatible token login, get an access token for future requests
    """
    user_service = UserService(session)
    user = user_service.authenticate(
        email=form_data.username, password=form_data.password
    )
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect email or password")
    elif not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    
    return {
        "access_token": create_access_token(user.email),
        "token_type": "bearer",
    }

@router.post("/login/json", response_model=Token)
def login_access_token_json(
    session: Annotated[Session, Depends(get_session)],
    login_data: UserLogin
) -> Any:
    """
    JSON body login, get an access token for future requests (easier for non-form clients)
    """
    user_service = UserService(session)
    user = user_service.authenticate(
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
def register_user(
    *,
    session: Annotated[Session, Depends(get_session)],
    user_in: UserCreate,
) -> Any:
    """
    Create new user.
    """
    user_service = UserService(session)
    user = user_service.get_by_email(email=user_in.email)
    if user:
        raise HTTPException(
            status_code=400,
            detail="The user with this email already exists in the system",
        )
    user = user_service.create_user(user_in=user_in)
    return user
