from typing import Optional, Union
from sqlmodel import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.schemas.settings import (
    SystemSettings, 
    SystemSettingsGeneralUpdate, 
    SystemSettingsDocumentsUpdate
)

class SettingsRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_settings(self) -> SystemSettings:
        """Get the global system settings. Create default row if it doesn't exist."""
        result = await self.session.execute(select(SystemSettings).where(SystemSettings.id == 1))
        settings = result.scalar_one_or_none()
        
        if not settings:
            settings = SystemSettings(id=1)
            self.session.add(settings)
            await self.session.commit()
            await self.session.refresh(settings)
            
        return settings

    async def update_settings(
        self, 
        update_data: Union[SystemSettingsGeneralUpdate, SystemSettingsDocumentsUpdate]
    ) -> SystemSettings:
        """Update the global system settings."""
        settings = await self.get_settings()
        
        update_dict = update_data.model_dump(exclude_unset=True)
        for key, value in update_dict.items():
            setattr(settings, key, value)
            
        self.session.add(settings)
        await self.session.commit()
        await self.session.refresh(settings)
        
        return settings
