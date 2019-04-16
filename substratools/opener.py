import abc
import inspect
import importlib
import sys
import types

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

    @abc.abstractmethod
    def get_pred(self, path):
        """Get predictions from path."""
        # TODO use a Serializer to define get/save pred?
        # FIXME should be consistent with load/save model from Algo
        raise NotImplementedError

    @abc.abstractmethod
    def save_pred(self, y_pred, path):
        """Save predictions to path."""
        # TODO use a Serializer to define get/save pred?
        raise NotImplementedError


REQUIRED_FUNCTIONS = set([
    'get_X', 'get_y', 'fake_X', 'fake_y', 'get_pred', 'save_pred'])


class OpenerWrapper(object):
    """Internal wrapper to call opener interface."""

    def __init__(self, opener):
        # validate opener
        if isinstance(opener, Opener):
            pass

        elif isinstance(opener, types.ModuleType):
            missing_functions = REQUIRED_FUNCTIONS.copy()
            for name, obj in inspect.getmembers(opener):
                if not inspect.isfunction(obj):
                    continue
                try:
                    missing_functions.remove(name)
                except KeyError:
                    pass

            if missing_functions:
                message = "Method(s) {} not implemented".format(
                    ", ".join(["'{}'".format(m) for m in missing_functions]))
                raise exceptions.InvalidOpener(message)

        else:
            raise exceptions.InvalidOpener(
                "Opener must be a module or an Opener instance")

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


def _load_interface_from_module(name):
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

    # backward compatibility; accept module
    return opener_module


def load_from_module(name='opener'):
    """Load opener interface based on current working directory.

    Opener can be defined as an Opener subclass or directly has a module.

    Return an OpenerWrapper instance.
    """
    interface = _load_interface_from_module(name)
    return OpenerWrapper(interface)
