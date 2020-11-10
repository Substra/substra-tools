import abc
import logging
import os
import types

from substratools import utils, exceptions
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

        def fake_X(self, n_samples):
            return []  # compute random fake data

        def fake_y(self, n_samples):
            return []  # compute random fake data

        def save_predictions(self, y_pred, path):
            with open(path, 'w') as fp:
                y_pred.to_csv(fp, index=False)

        def get_predictions(self, path):
            return pd.read_csv(path)
    ```

    # How to test locally an opener script

    An opener can be imported and used in python scripts as would any other class.

    For example, assuming that you have a local file named `opener.py` that contains
    an `Opener` named  `MyOpener`:

    ```python
    import os
    from opener import MyOpener

    folders = os.listdir('./sandbox/data_samples/')

    o = MyOpener()
    X = o.get_X(folders)
    y = o.get_y(folders)
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
    def fake_X(self, n_samples):
        """Generate a fake matrix of features for offline testing.

        # Arguments

        n_samples (int): number of samples to return

        # Returns

        data: data labels object.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def fake_y(self, n_samples):
        """Generate a fake target variable vector for offline testing.

        # Arguments

        n_samples (int): number of samples to return

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
        return self._workspace.input_data_folder_paths

    def get_X(self, fake_data=False, n_fake_samples=None):
        if fake_data:
            logger.info("loading X from fake data")
            return self._interface.fake_X(n_samples=n_fake_samples)
        else:
            logger.info("loading X from '{}'".format(self.data_folder_paths))
            return self._interface.get_X(self.data_folder_paths)

    def get_y(self, fake_data=False, n_fake_samples=None):
        if fake_data:
            logger.info("loading y from fake data")
            return self._interface.fake_y(n_samples=n_fake_samples)
        else:
            logger.info("loading y from '{}'".format(self.data_folder_paths))
            return self._interface.get_y(self.data_folder_paths)

    def get_predictions(self):
        path = self._workspace.input_predictions_path
        logger.info("loading predictions from '{}'".format(path))
        return self._interface.get_predictions(path)

    def _assert_predictions_file_exists(self):
        path = self._workspace.output_predictions_path
        if os.path.isdir(path):
            raise exceptions.NotAFileError(f'Expected predictions file at {path}, found dir')
        if not os.path.isfile(path):
            raise exceptions.MissingFileError(f'Predictions file {path} does not exists')

    def save_predictions(self, y_pred):
        path = self._workspace.output_predictions_path
        logger.info("saving predictions to '{}'".format(path))
        res = self._interface.save_predictions(y_pred, path)
        self._assert_predictions_file_exists()
        return res


def load_from_module(path=None, workspace=None):
    """Load opener interface from path or from python environment.

    Opener can be defined as an Opener subclass or directly has a module.

    Return an OpenerWrapper instance.
    """
    interface = utils.load_interface_from_module(
        'opener',
        interface_class=Opener,
        interface_signature=None,  # XXX does not support interface for debugging
        path=path,
    )
    return OpenerWrapper(interface, workspace=workspace)
