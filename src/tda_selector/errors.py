class SelectorError(Exception):
    """Base exception for selector failures."""


class ValidationError(SelectorError):
    """Raised for invalid user inputs."""


class CacheError(SelectorError):
    """Raised for cache incompatibility or corruption."""
