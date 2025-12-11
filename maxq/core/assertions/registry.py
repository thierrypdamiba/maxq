"""Assertion registry for dynamic assertion lookup."""

from typing import Any

from maxq.core.assertions.base import Assertion

# Global registry of assertion types
_ASSERTION_REGISTRY: dict[str, type[Assertion]] = {}


def register_assertion(name: str):
    """
    Decorator to register an assertion type.

    Usage:
        @register_assertion("my-assertion")
        class MyAssertion(Assertion):
            ...
    """
    def decorator(cls: type[Assertion]) -> type[Assertion]:
        cls.assertion_type = name
        _ASSERTION_REGISTRY[name] = cls
        return cls
    return decorator


def get_assertion(name: str, config: dict[str, Any]) -> Assertion:
    """
    Get an assertion instance by name.

    Args:
        name: Assertion type name (e.g., "contains-id", "ndcg")
        config: Configuration dict for the assertion

    Returns:
        Configured Assertion instance

    Raises:
        ValueError: If assertion type is unknown
    """
    if name not in _ASSERTION_REGISTRY:
        available = ", ".join(sorted(_ASSERTION_REGISTRY.keys()))
        raise ValueError(
            f"Unknown assertion type: '{name}'. "
            f"Available types: {available}"
        )
    return _ASSERTION_REGISTRY[name](config)


def list_assertions() -> list[str]:
    """List all registered assertion types."""
    return sorted(_ASSERTION_REGISTRY.keys())
