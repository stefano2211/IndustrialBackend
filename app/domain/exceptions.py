"""
Domain exceptions for the Industrial Backend.

These exceptions are raised by the service layer and caught by the API layer.
This keeps the domain layer clean of HTTP/framework concerns (DIP).
"""


class DomainError(Exception):
    """Base exception for all domain errors."""
    pass


class NotFoundError(DomainError):
    """Raised when a requested resource does not exist."""

    def __init__(self, resource: str, identifier=None):
        self.resource = resource
        self.identifier = identifier
        detail = f"{resource} not found"
        if identifier:
            detail = f"{resource} '{identifier}' not found"
        super().__init__(detail)


class AccessDeniedError(DomainError):
    """Raised when a user tries to access a resource they don't own."""

    def __init__(self, resource: str = "resource"):
        super().__init__(f"Access denied to {resource}")


class ValidationError(DomainError):
    """Raised when business validation fails."""

    def __init__(self, detail: str):
        super().__init__(detail)
