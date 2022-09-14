import abc
import logging
import os
import types
from typing import Optional

from substratools import exceptions
from substratools import utils
from substratools.workspace import OpenerWorkspace

logger = logging.getLogger(__name__)


REQUIRED_FUNCTIONS = set(
    [
        "get_X",
        "get_y",
        "fake_X",
        "fake_y",
    ]
)


class Opener(abc.ABC):
    """Dataset opener abstract base class.

    To define a new opener script, subclass this class and implement the
    following abstract methods:

    - #Opener.get_X()
    - #Opener.get_y()
    - #Opener.fake_X()
    - #Opener.fake_y()

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


class OpenerWrapper(object):
    """Internal wrapper to call opener interface."""

    def __init__(self, interface, workspace=None):
        assert isinstance(interface, Opener) or isinstance(interface, types.ModuleType)

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

    def _assert_output_exists(self, path, key):

        if os.path.isdir(path):
            raise exceptions.NotAFileError(f"Expected output file at {path}, found dir for output `{key}`")
        if not os.path.isfile(path):
            raise exceptions.MissingFileError(f"Output file {path} used to save argument `{key}` does not exists.")


def load_from_module(workspace=None) -> Optional[OpenerWrapper]:
    """Load opener interface.

    If a workspace is given, the associated opener will be returned. This means that if no
    opener_path is defined within the workspace, no opener will be returned
    If no workspace is given, the opener interface will be directly loaded as a module.

    Return an OpenerWrapper instance.
    """
    if workspace is None:
        # import from module
        path = None

    elif workspace.opener_path is None:
        # no opener within this workspace
        return None

    else:
        # import opener from workspace specified path
        path = workspace.opener_path

    interface = utils.load_interface_from_module(
        "opener",
        interface_class=Opener,
        interface_signature=None,  # XXX does not support interface for debugging
        path=path,
    )
    return OpenerWrapper(interface, workspace=workspace)
