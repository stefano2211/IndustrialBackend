from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select
from app.domain.proactiva.schemas.model import Model, ModelUpdate

class ModelRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_id(self, model_id: str) -> Optional[Model]:
        return await self.session.get(Model, model_id)

    async def list_all(self) -> List[Model]:
        statement = select(Model)
        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def create(self, model: Model) -> Model:
        self.session.add(model)
        await self.session.commit()
        await self.session.refresh(model)
        return model

    async def update(self, model_id: str, update_data: ModelUpdate) -> Optional[Model]:
        model = await self.get_by_id(model_id)
        if not model:
            return None
        
        data = update_data.model_dump(exclude_unset=True)
        for key, value in data.items():
            setattr(model, key, value)
            
        self.session.add(model)
        await self.session.commit()
        await self.session.refresh(model)
        return model

    async def delete(self, model_id: str) -> bool:
        model = await self.get_by_id(model_id)
        if not model:
            return False
        
        await self.session.delete(model)
        await self.session.commit()
        return True
