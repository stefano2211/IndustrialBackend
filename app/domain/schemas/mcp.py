from pydantic import BaseModel, Field
from typing import List, Optional, Any, Union

class KeyFigure(BaseModel):
    """Represents a numerical metric (hot path data)."""
    name: str = Field(..., description="Name of the metric (e.g., 'Temperature')")
    value: float = Field(..., description="Numerical value of the metric")
    unit: Optional[str] = Field(None, description="Measurement unit (e.g., '°C')")
    timestamp: Optional[str] = Field(None, description="ISO timestamp of the reading")

class KeyValue(BaseModel):
    """Represents a categorical/status value (non-numerical data)."""
    name: str = Field(..., description="Name of the field (e.g., 'Status')")
    value: Any = Field(..., description="Categorical or descriptive value")
    metadata: Optional[dict] = Field(None, description="Additional context")

class MCPResponse(BaseModel):
    """Standardized response container for any MCP source."""
    source: str = Field(..., description="Name of the data source")
    key_figures: List[KeyFigure] = Field(default_factory=list)
    key_values: List[KeyValue] = Field(default_factory=list)
    raw_response: Optional[dict] = Field(None, description="Original payload for troubleshooting")
    error: Optional[str] = None
