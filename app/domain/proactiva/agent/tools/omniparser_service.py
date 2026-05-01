"""Re-export wrapper --- implementation moved to app.domain.shared.agent.tools.omniparser_service."""
from app.domain.shared.agent.tools.omniparser_service import (
    ParsedElement,
    OmniParserResult,
    OmniParserService,
    get_omniparser,
)

__all__ = [
    "ParsedElement",
    "OmniParserResult",
    "OmniParserService",
    "get_omniparser",
]
