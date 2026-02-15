from typing import Generator, Annotated
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from pydantic import ValidationError
from sqlmodel import Session
from app.core.database import get_session
from app.core.security import ALGORITHM
from app.config import settings
from app.domain.models.user import User
from app.domain.schemas.token import TokenPayload
from app.domain.services.user_service import UserService

security = HTTPBearer()

def get_current_user(
    session: Annotated[Session, Depends(get_session)],
    token: Annotated[HTTPAuthorizationCredentials, Depends(security)]
) -> User:
    try:
        payload = jwt.decode(
            token.credentials, settings.secret_key, algorithms=[ALGORITHM]
        )
        token_data = TokenPayload(**payload)
    except (JWTError, ValidationError):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials",
        )
    
    user_service = UserService(session)
    # token_data.sub is stored as string in token
    user = user_service.get_by_email(email=token_data.sub)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return user
