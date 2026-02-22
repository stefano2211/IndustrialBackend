from sqlmodel import SQLModel, Session, create_engine
from app.core.config import settings
from typing import Generator

# Sync Database URL (using psycopg2 by default for 'postgresql://')
DATABASE_URL = f"postgresql://{settings.postgres_user}:{settings.postgres_password}@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"

engine = create_engine(DATABASE_URL, echo=False)

def get_session() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session
