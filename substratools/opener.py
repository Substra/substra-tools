import abc
import logging
import os
import types

from substratools import utils
from substratools.workspace import OpenerWorkspace

logger = logging.getLogger(__name__)


REQUIRED_FUNCTIONS = set([
    'get_X',
    'get_y',
    'fake_X',
    'fake_y',
    'get_predictions',
    'save_predictions',
])


class Opener(abc.ABC):
    """Dataset opener abstract base class.

    To define a new opener script, subclass this class and implement the
    following abstract methods:

    - #Opener.get_X()
    - #Opener.get_y()
    - #Opener.fake_X()
    - #Opener.fake_y()
    - #Opener.get_predictions()
    - #Opener.save_predictions()

    # Example

    ```python
    import os
    import pandas as pd
    import string
    import numpy as np

    import substratools as tools

    class DummyOpener(tools.Opener):
        def get_X(self, folders):
            return [
                pd.read_csv(os.path.join(folder, 'train.csv'))
                for folder in folders
            ]

        def get_y(self, folders):
            return [
                pd.read_csv(os.path.join(folder, 'y.csv'))
                for folder in folders
            ]

        def fake_X(self):
            return []  # compute random fake data

        def fake_y(self):
            return []  # compute random fake data

        def save_predictions(self, y_pred, path):
            with open(path, 'w') as fp:
                y_pred.to_csv(fp, index=False)

        def get_predictions(self, path):
            return pd.read_csv(path)
    ```
    """

    @abc.abstractmethod
    def get_X(self, folders):
        """Load feature data from data sample folders.

        # Arguments

        folders: list of folders. Each folder represents a data sample.

        # Returns

        data: data object.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_y(self, folders):
        """Load labels from data sample folders.

        # Arguments

        folders: list of folders. Each folder represents a data sample.

        # Returns

        data: data labels object.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def fake_X(self):
        """Generate a fake matrix of features for offline testing.

        # Returns

        data: data labels object.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def fake_y(self):
        """Generate a fake target variable vector for offline testing.

        # Returns

        data: data labels object.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_predictions(self, path):
        """Read file and return predictions vector.

        # Arguments

        path: string file path.

        # Returns

        predictions: predictions vector.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def save_predictions(self, y_pred, path):
        """Write predictions vector to file.

        # Arguments

        y_pred: predictions vector.
        path: string file path.
        """
        raise NotImplementedError


class OpenerWrapper(object):
    """Internal wrapper to call opener interface."""

    def __init__(self, interface, workspace=None):
        assert isinstance(interface, Opener) or \
            isinstance(interface, types.ModuleType)

        self._workspace = workspace or OpenerWorkspace()
        self._interface = interface

    @property
    def data_folder_paths(self):
        rootpath = self._workspace.input_data_folder_path
        folders = [os.path.join(rootpath, subfolder_name)
                   for subfolder_name in os.listdir(rootpath)
                   if os.path.isdir(os.path.join(rootpath, subfolder_name))]
        return folders

    def get_X(self, fake_data=False):
        if fake_data:
            logger.info("loading X from fake data")
            return self._interface.fake_X()
        else:
            logger.info("loading X from '{}'".format(self.data_folder_paths))
            return self._interface.get_X(self.data_folder_paths)

    def get_y(self, fake_data=False):
        if fake_data:
            logger.info("loading y from fake data")
            return self._interface.fake_y()
        else:
            logger.info("loading y from '{}'".format(self.data_folder_paths))
            return self._interface.get_y(self.data_folder_paths)

    def get_predictions(self):
        path = self._workspace.input_predictions_path
        logger.info("loading predictions from '{}'".format(path))
        return self._interface.get_predictions(path)

    def save_predictions(self, y_pred):
        path = self._workspace.output_predictions_path
        logger.info("saving predictions to '{}'".format(path))
        return self._interface.save_predictions(y_pred, path)


def load_from_module(path=None, workspace=None):
    """Load opener interface from path or from python environment.

    Opener can be defined as an Opener subclass or directly has a module.

    Return an OpenerWrapper instance.
    """
    interface = utils.load_interface_from_module(
        'opener',
        interface_class=Opener,
        interface_signature=REQUIRED_FUNCTIONS,
        path=path,
    )
    return OpenerWrapper(interface, workspace=workspace)
