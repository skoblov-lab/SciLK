"""

This module is purely experimental. We are going to use it for IO prototyping.

"""


from typing import Tuple, Mapping, List, Optional, Callable, Any
from importlib import import_module
import joblib
import inspect
import keyword
import copy
import glob
import shutil
import os

import scilk


LOADER_EXT = 'loader'
DATA_EXT = 'entrydata'
COLL_EXT = 'collection'


class Collection:

    def __init__(self):
        self._entries = {}
        self._loaders = {}
        self._data = {}
        self._status = {}

    def __getattr__(self, entry: str) -> Any:
        if entry not in self._loaders:
            raise AttributeError('no entry named {}'.format(entry))
        # uninvoked dependencies are False, loading dependencies are None,
        # loaded dependencies are True
        if self._status[entry] is None:
            raise RuntimeError("'{}' was accessed while loading".format(entry))
        if not self._status[entry]:
            self._activate_entry(entry)
        return self._entries[entry]

    @property
    def entries(self) -> List[str]:
        return list(self._loaders)

    def add(self, entry: str, loader: Callable[['Collection', Mapping], Any],
            data: Optional[Mapping[str, str]]=None, postpone: bool=False):
        """
        Add a model to the collection.
        :param entry: entry name; it must be a valid python identifier,
        because it will be used to access the entry via the attribute lookup
        mechanism, i.e.
        >>> assert isidentifier(entry)
        should pass.
        :param loader: a callable responsible for loading an entry. A loader
        must accept two arguments: (1) a Collection instance (this would allow
        the loader to access other models in the same Collection) and (2) a data
        mapping (see argument 'data'); take note that cyclic dependencies
        between entries are not allowed and will result in a RuntimeError
        error. There are two additional requirements:
        - The loader must be defined in an importable module
        - The loader must be accessible via its __name__ attribute from the
        module's global namespace.
        If both requirements are met, the following code will work just fine:
        >>> import inspect
        >>> from importlib import import_module
        >>> module = import_module(inspect.getmodule(loader).__name__)
        >>> assert getattr(module, loader.__name__) is loader
        The method will try to validate your loader and will raise a ValueError
        if the validation fails.
        :param data: a Mapping between labels and file paths (symlinks are not
        allowed). When a Collection is serialised, all data mappings associated
        with underlying entries are copied into the Collection's destination
        directory under appropriate subdirectories; nevertheless, all data keys
        remain the same and it is thus safe to rely on them in loaders.
        :param postpone: do not load the entry at once. This option is useful if
        you don't want to work out the correct order of adding entries without
        running into missing dependencies.
        :raises SyntaxError: invalid entry name
        :raises ImportError: can't import loader from its module
        :raises ValueError: invalid data
        """
        if not isidentifier(entry):
            raise SyntaxError("'{}' is not a valid identifier".format(entry))
        if not importable(loader):
            raise ImportError("can't import the loader from its module")
        # check the data mapping
        if not (data is None or isinstance(data, Mapping)):
            raise ValueError('data argument must be a Mapping instance or None')
        if not (data is None or all(map(os.path.isfile, data.values()))):
            raise ValueError('all values in data must be valid file paths')
        self._loaders[entry] = loader
        self._data[entry] = copy.deepcopy(dict(data or {}))
        self._status[entry] = False
        if not postpone:
            self._activate_entry(entry)

    def _activate_entry(self, entry: str):
        if self._status[entry]:
            raise RuntimeError('trying to reload an entry')
        # set entry status to None to show that it is currently loading
        self._status[entry] = None
        # load the entry
        self._entries[entry] = self._loaders[entry](self, self._data[entry])
        # show that the entry is available
        self._status[entry] = True

    @classmethod
    def load(cls, name: str) -> 'Collection':
        """
        Load a serialised Collection from your SciLK root
        :param name: Collection's name
        :return: a loaded Collection
        :raises FileNotFoundError: missing files
        :raises ModuleNotFoundError: can't load a loader's module
        :raises AttributeError: can't find a loader in its module
        """
        collection = cls()
        base = os.path.join(scilk.SCILK_ROOT, name)
        entries = joblib.load(os.path.join(base, '{}.{}'.format(name, COLL_EXT)))
        for entry in entries:
            collection.add(entry, *cls._load_entry(base, entry), postpone=True)
        return collection

    def save(self, name):
        """
        Save a Collection to your SciLK root in a distributable form:
        - create a directory named after the Collection under the SciLK root
        directory and inflate it with subdirectories named after entries
        - save everything necessary to load the entries
        - save specifications
        :raises FileExistsError: there already is a saved Collection with
        identical name
        """
        destination = os.path.join(scilk.SCILK_ROOT, name)
        try:
            os.makedirs(destination)
        except FileExistsError:
            raise FileExistsError("there is a collection named '{}' in your "
                                  "SciLK root directory".format(name))
        # save individual entries
        for entry in self._loaders:
            self._save_entry(destination, entry)
        # save collection spec to prevent data corruption
        collection_spec_path = os.path.join(destination,
                                            '{}.{}'.format(name, COLL_EXT))
        joblib.dump(self.entries, collection_spec_path, 1)

    @staticmethod
    def _load_entry(base: str, entry: str) -> Tuple[Callable, Mapping]:
        # load data
        data_spec_path = os.path.join(base, entry, '{}.{}'.format(entry, DATA_EXT))
        try:
            data_spec = joblib.load(data_spec_path)
        except FileNotFoundError:
            raise FileNotFoundError("missing data for entry '{}'".format(entry))
        data = {k: os.path.join(base, entry, value) for k, value in data_spec.items()}
        # load loader
        loader_spec_path = os.path.join(base, entry, '{}.{}'.format(entry, LOADER_EXT))
        try:
            module, name = joblib.load(loader_spec_path)
        except FileNotFoundError:
            raise FileNotFoundError("missing loader for entry '{}'".format(entry))
        try:
            loader = getattr(import_module(module), name)
        except ModuleNotFoundError:
            raise ModuleNotFoundError("can't import module '{}' to access "
                                      "the loader specified by "
                                      "'{}'".format(module, entry))
        except AttributeError:
            raise AttributeError("module '{}' contains no global name "
                                 "'{}' specified as loader in entry "
                                 "'{}'".format(module, name, entry))
        return loader, data

    def _save_entry(self, base, entry):
        destination = os.path.join(base, entry)
        os.mkdir(destination)
        # save data and data spec
        data = self._data[entry]
        for _, path in data.items():
            shutil.copy(path, os.path.join(destination, os.path.basename(path)))
        data_spec = {item: os.path.basename(path) for item, path in data.items()}
        data_spec_path = os.path.join(destination, '{}.{}'.format(entry, DATA_EXT))
        joblib.dump(data_spec, data_spec_path, 1)
        # save loader spec
        loader = self._loaders[entry]
        loader_spec = (inspect.getmodule(loader).__name__, loader.__name__)
        loader_spec_path = os.path.join(destination, '{}.{}'.format(entry, LOADER_EXT))
        joblib.dump(loader_spec, loader_spec_path, 1)


def importable(item) -> bool:
    """
    Check whether 'item' is accessible from its module's global namespace under
    'item.__name__'.
    :param item:
    :return:
    """
    try:
        module = import_module(inspect.getmodule(item).__name__)
        assert getattr(module, item.__name__) is item
    except (AssertionError, ImportError, ValueError, AttributeError):
        return False
    return True


def isidentifier(name: str) -> bool:
    """
    Determines if string is valid Python identifier.
    """
    if not isinstance(name, str):
        raise TypeError("expected str, but got {!r}".format(type(name)))
    return name.isidentifier() and not keyword.iskeyword(name)


if __name__ == '__main__':
    raise RuntimeError
