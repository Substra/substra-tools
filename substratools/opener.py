import abc
import inspect
import importlib
import sys

from substratools import exceptions, workspace


class Opener(abc.ABC):
    """Dataset opener abstract base class."""

    @abc.abstractmethod
    def get_X(self, folder):
        """Load feature data from folder."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_y(self, folder):
        """Load labels."""
        raise NotImplementedError

    @abc.abstractmethod
    def fake_X(self):
        raise NotImplementedError

    @abc.abstractmethod
    def fake_y(self):
        raise NotImplementedError

    def get_pred(self, path):
        """Get predictions from path."""
        raise NotImplementedError

    def save_pred(self, y_pred, path):
        """Save predictions to path."""
        raise NotImplementedError


class OpenerWrapper(object):
    """Internal wrapper to call opener interface."""

    def __init__(self, opener):
        self._interface = opener
        self._workspace = workspace.Workspace()

    def get_X(self, dry_run=False):
        if dry_run:
            return self._interface.fake_X()
        else:
            return self._interface.get_X(self._workspace.data_folder)

    def get_y(self, dry_run=False):
        if dry_run:
            return self._interface.fake_y()
        else:
            return self._interface.get_y(self._workspace.data_folder)

    def get_pred(self):
        return self._interface.get_pred(self._workspace.pred_filepath)

    def save_pred(self, y_pred):
        return self._interface.save_pred(y_pred, self._workspace.pred_filepath)


def load_interface_from_module(name):
    try:
        del sys.modules[name]
    except KeyError:
        pass

    try:
        opener_module = importlib.import_module(name)
    except ModuleNotFoundError:
        raise exceptions.OpenerModuleNotFound()

    # check if opener has an Opener class
    for name, obj in inspect.getmembers(opener_module):
        if inspect.isclass(obj) and issubclass(obj, Opener):
            opener_class = obj
            o = opener_class()
            return o

    # backward compatibility; ensure module has the following methods
    required_methods = set(['get_X', 'get_y', 'fake_X', 'fake_y', 'get_pred',
                            'save_pred'])
    for name, obj in inspect.getmembers(opener_module):
        if not inspect.isfunction(obj):
            continue
        try:
            required_methods.remove(name)
        except KeyError:
            pass

    if required_methods:
        raise exceptions.InvalidOpener("Method(s) {} not implemented".format(
            ", ".join(["'{}'".format(m) for m in required_methods])))
    return opener_module


def load_from_module(name='opener'):
    """Load opener interface based on current working directory.

    Opener can be defined as an Opener subclass or directly has a module.
    """
    interface = load_interface_from_module(name)
    return OpenerWrapper(interface)
