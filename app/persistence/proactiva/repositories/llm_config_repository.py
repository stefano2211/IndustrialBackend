from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select
from app.domain.schemas.llm_config import LLMConfig, LLMConfigUpdate

class LLMConfigRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_config(self, role: str) -> Optional[LLMConfig]:
        return await self.session.get(LLMConfig, role)

    async def list_configs(self) -> List[LLMConfig]:
        statement = select(LLMConfig)
        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def set_config(self, role: str, provider: str, model_name: str) -> LLMConfig:
        config = await self.get_config(role)
        if config:
            config.provider = provider
            config.model_name = model_name
        else:
            config = LLMConfig(role=role, provider=provider, model_name=model_name)
        
        self.session.add(config)
        await self.session.commit()
        await self.session.refresh(config)
        return config

    async def update_config(self, role: str, update_data: LLMConfigUpdate) -> Optional[LLMConfig]:
        config = await self.get_config(role)
        if not config:
            return None
        
        data = update_data.model_dump(exclude_unset=True)
        for key, value in data.items():
            setattr(config, key, value)
            
        self.session.add(config)
        await self.session.commit()
        await self.session.refresh(config)
        return config
