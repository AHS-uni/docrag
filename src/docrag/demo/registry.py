"""DocRAG demo – generic registry for plug-in backends."""

from typing import Callable, Type, TypeVar

T = TypeVar("T")


class Registry:
    """Light-weight keyed registry for classes and singletons."""

    def __init__(self):
        self._classes: dict[str, Type[T]] = {}
        self._instances: dict[str, T] = {}

    def register(self, key: str) -> Callable[[Type[T]], Type[T]]:
        """Decorator to register a class under the given key.

        Args:
            key: Unique name for the class in this registry.
        """

        def decorator(cls: Type[T]) -> Type[T]:
            k = key.lower()
            if k in self._classes:
                raise ValueError(f"Duplicate key '{k}'")
            self._classes[k] = cls
            return cls

        return decorator

    def available(self) -> list[str]:
        """List all registered keys.

        Returns:
            Sorted list of registry keys.
        """
        return sorted(self._classes.keys())

    def load(self, key: str, *args, **kwargs) -> T:
        """Instantiate and cache the class for the given key.

        Args:
            key: Registry key.
            *args, **kwargs: Passed to class constructor.

        Returns:
            Singleton instance of the registered class.
        """
        k = key.lower()
        if k not in self._instances:
            cls = self._classes[k]
            self._instances[k] = cls(*args, **kwargs)
        return self._instances[k]

    def get(self, key: str) -> T:
        """Get the already loaded instance for key.

        Args:
            key: Registry key.

        Returns:
            The cached instance.

        Raises:
            KeyError: if not loaded yet.
        """
        return self._instances[key.lower()]

    def unload(self, key: str) -> None:
        """Remove the cached instance, freeing memory if needed."""
        self._instances.pop(key.lower(), None)

    def loaded(self) -> list[str]:
        """List all keys for which an instance is loaded."""
        return list(self._instances.keys())


# Two registries for this demo (retrievers & future generators)
RETRIEVERS = Registry()
GENERATORS = Registry()
