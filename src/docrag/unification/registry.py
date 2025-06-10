__all__ = ["register_unifier", "get_unifier"]

_UNIFIER_REGISTRY: dict[str, type] = {}

def register_unifier(name: str):
    """
    Decorator to register a Unifier subclass under a given dataset name.
    Usage:

        @register_unifier("tatdqa")
        class TATDQAUnifier(Unifier[TATDQARawEntry]):
            ...
    """

    def decorator(cls):
        _UNIFIER_REGISTRY[name] = cls
        return cls

    return decorator


def get_unifier(name: str) -> type | None:
    """
    Return the Unifier subclass registered under `name`, or None if not found.
    """
    return _UNIFIER_REGISTRY.get(name)
