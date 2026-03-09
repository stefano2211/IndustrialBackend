import uuid
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select
from app.domain.schemas.prompt import Prompt, PromptUpdate

class PromptRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_prompt(self, prompt: Prompt) -> Prompt:
        self.session.add(prompt)
        await self.session.commit()
        await self.session.refresh(prompt)
        return prompt

    async def get_prompt(self, prompt_id: uuid.UUID) -> Optional[Prompt]:
        return await self.session.get(Prompt, prompt_id)

    async def list_prompts(self, only_enabled: bool = False) -> List[Prompt]:
        statement = select(Prompt)
        if only_enabled:
            statement = statement.where(Prompt.is_enabled == True)
        result = await self.session.execute(statement)
        return list(result.scalars().all())

    async def update_prompt(self, prompt_id: uuid.UUID, prompt_update: PromptUpdate) -> Optional[Prompt]:
        db_prompt = await self.get_prompt(prompt_id)
        if not db_prompt:
            return None
        
        prompt_data = prompt_update.model_dump(exclude_unset=True)
        for key, value in prompt_data.items():
            setattr(db_prompt, key, value)
        
        self.session.add(db_prompt)
        await self.session.commit()
        await self.session.refresh(db_prompt)
        return db_prompt

    async def delete_prompt(self, prompt_id: uuid.UUID) -> bool:
        db_prompt = await self.get_prompt(prompt_id)
        if not db_prompt:
            return False
        await self.session.delete(db_prompt)
        await self.session.commit()
        return True
