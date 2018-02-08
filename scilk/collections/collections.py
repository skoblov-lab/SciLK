"""



"""
from typing import Sequence, Mapping, Optional, Callable, Any


class DependencyError(RuntimeError):
    pass


class Collection:

    def __init__(self, name: str):
        """

        :param name:
        """
        self._name = name
        self._models = {}
        self._data = {}
        self._status = {}

    def __getattr__(self, entry: str) -> Any:
        pass

    def add(self, name: str, loader: Callable[['Collection', Mapping], Any],
            data: Optional[Mapping[str, str]]=None):
        """
        Add a model to the collection.
        :param name: entry name; it must be a valid python attribute name,
        because it will be used to access the entry via the attribute lookup
        mechanism.
        :param loader: a callable responsible for loading an entry. A loader
        must accept two arguments: (1) a Collection instance (this would allow
        the loader to access other models in the same Collection) and (2) a data
        mapping (see argument 'data'); take note that cyclic dependencies
        between entries are not allowed and will result in a DependencyError
        error. If you ever plan to serialise a Collection with this loader on
        board, the loader must meet the following requirements:
        - The loader must be defined in an importable module
        - The loader must be accessible via its __name__ attribute from the
        module's global namespace.
        If both requirements are met, the following code will work just fine:
        >>> import inspect
        >>> from importlib import import_module
        >>> module = import_module(inspect.getmodule(loader))
        >>> assert getattr(module, loader.__name__) is loader
        :param data: a Mapping between labels and file paths (symlinks are not
        allowed). When a Collection is serialised, all data mappings associated
        with underlying entries are copied into the Collection's destination
        directory under appropriate subdirectories; nevertheless, all data keys
        remain the same and it is thus safe to rely on them in loaders.
        :raises NameError: invalid entry name
        :raises DependencyError: missing data or missing/circular dependencies
        """
        pass

    def _validate_loader(self, loader: Callable[['Collection', Mapping], Any]) -> bool:
        pass

    def _load(self, entry: str):
        pass

    def save(self, root: str=None):
        pass


if __name__ == '__main__':
    raise RuntimeError
