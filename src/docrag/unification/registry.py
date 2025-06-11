from typing import Type
from .unifier import Unifier

__all__ = ["register", "get_unifier"]

_registry: dict[str, type] = {}


def register(name: str):
    """
    Decorator to register a Unifier subclass under a given dataset name.
    Matching is case-insensitive
    """

    def decorator(cls: Type[Unifier]) -> Type[Unifier]:
        _registry[name.lower()] = cls
        return cls

    return decorator


def get_unifier(name: str) -> Type[Unifier]:
    """
    Lookup a Unifier by key.
    Raises ValueError if no matching unifier is found.
    """
    key = name.lower()
    if key in _registry:
        return _registry[key]
    available = ", ".join(_registry.keys())
    raise ValueError(
        f"No unifier registered for '{name}'. Available unifiers: {available}"
    )
