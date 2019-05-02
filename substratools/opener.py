import abc
import os
import types

from substratools import workspace, utils


REQUIRED_FUNCTIONS = set([
    'get_X', 'get_y', 'fake_X', 'fake_y', 'get_pred', 'save_pred'])


class Opener(abc.ABC):
    """Dataset opener abstract base class."""

    @abc.abstractmethod
    def get_X(self, folders):
        """Load feature data from data sample folders."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_y(self, folders):
        """Load labels from data sample folders."""
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
        raise NotImplementedError

    @abc.abstractmethod
    def save_pred(self, y_pred, path):
        """Save predictions to path."""
        raise NotImplementedError


class OpenerWrapper(object):
    """Internal wrapper to call opener interface."""

    def __init__(self, opener):
        assert isinstance(opener, Opener) or \
            isinstance(opener, types.ModuleType)

        self._interface = opener
        self._workspace = workspace.Workspace()

    @property
    def data_folder_paths(self):
        rootpath = self._workspace.data_folder
        folders = [os.path.join(rootpath, subfolder)
                   for subfolder in os.listdir(rootpath)]
        return folders

    def get_X(self, dry_run=False):
        if dry_run:
            return self._interface.fake_X()
        else:
            return self._interface.get_X(self.data_folder_paths)

    def get_y(self, dry_run=False):
        if dry_run:
            return self._interface.fake_y()
        else:
            return self._interface.get_y(self.data_folder_paths)

    def get_pred(self):
        return self._interface.get_pred(self._workspace.pred_filepath)

    def save_pred(self, y_pred):
        return self._interface.save_pred(y_pred, self._workspace.pred_filepath)


def load_from_module(name='opener'):
    """Load opener interface based on current working directory.

    Opener can be defined as an Opener subclass or directly has a module.

    Return an OpenerWrapper instance.
    """
    interface = utils.load_interface_from_module(
        name,
        interface_class=Opener,
        interface_signature=REQUIRED_FUNCTIONS)
    return OpenerWrapper(interface)
