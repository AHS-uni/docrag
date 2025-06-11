from typing import Type
from .adapter import Adapter

__all__ = ["register", "get_adapter"]

_registry: dict[str, Type[Adapter]] = {}


def register(name: str):
    """
    Decorator to register an Adapter under a given key.
    Matching is case‐insensitive.
    """

    def decorator(cls: Type[Adapter]) -> Type[Adapter]:
        _registry[name.lower()] = cls
        return cls

    return decorator


def get_adapter(name: str) -> Type[Adapter]:
    """
    Lookup an Adapter by (HF) model name or key.
    Raises ValueError if no matching adapter is found.
    """
    key = name.lower()
    if key in _registry:
        return _registry[key]
    available = ", ".join(_registry.keys())
    raise ValueError(
        f"No adapter registered for '{name}'. Available adapters: {available}"
    )
