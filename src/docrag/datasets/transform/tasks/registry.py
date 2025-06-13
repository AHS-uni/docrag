from collections.abc import Callable

__all__ = ["register", "get_task_transform"]

_registry: dict[str, Callable] = {}


def register(name: str):
    """
    Decorator to register a transform task under a given key.
    Matching is case‐insensitive.
    """

    def decorator(function: Callable) -> Callable:
        _registry[name.lower()] = function
        return function

    return decorator


def get_task_transform(name: str) -> Callable:
    """
    Lookup an transform task by  key.
    Raises ValueError if no matching implementation is found.
    """
    key = name.lower()
    if key in _registry:
        return _registry[key]
    available = ", ".join(_registry.keys())
    raise ValueError(
        f"No transform task registered for '{name}'. Available task transforms: {available}"
    )
