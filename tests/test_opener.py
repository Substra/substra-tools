import os

import pytest

from substratools import exceptions
from substratools.opener import load_from_module
from substratools.utils import import_module


@pytest.fixture
def tmp_cwd(tmp_path):
    # create a temporary current working directory
    new_dir = tmp_path / "workspace"
    new_dir.mkdir()

    old_dir = os.getcwd()
    os.chdir(new_dir)

    yield new_dir

    os.chdir(old_dir)


def test_load_opener_not_found(tmp_cwd):
    with pytest.raises(ImportError):
        load_from_module()


def test_load_invalid_opener(tmp_cwd):
    invalid_script = """
def get_X():
    raise NotImplementedError
def get_y():
    raise NotImplementedError
"""

    import_module('opener', invalid_script)

    with pytest.raises(exceptions.InvalidInterface):
        load_from_module()


def test_load_opener_as_module(tmp_cwd):
    script = """
def _helper():
    pass
def get_X(folders):
    return 'X'
def get_y(folders):
    return 'y'
def fake_X():
    return 'fakeX'
def fake_y():
    return 'fakey'
def get_pred(path):
    return 'pred'
def save_pred(y_pred, path):
    return 'pred'
"""

    import_module('opener', script)

    o = load_from_module()
    assert o.get_X() == 'X'


def test_load_opener_as_class(tmp_cwd):
    script = """
from substratools import Opener
class MyOpener(Opener):
    def get_X(self, folders):
        return 'Xclass'
    def get_y(self, folders):
        return 'yclass'
    def fake_X(self):
        return 'fakeX'
    def fake_y(self):
        return 'fakey'
    def get_pred(self, path):
        return 'pred'
    def save_pred(self, y_pred, path):
        return 'pred'
"""

    import_module('opener', script)

    o = load_from_module()
    assert o.get_X() == 'Xclass'


def test_opener_check_folders(tmp_cwd):
    script = """
from substratools import Opener
class MyOpener(Opener):
    def get_X(self, folders):
        assert len(folders) == 5
        return 'Xclass'
    def get_y(self, folders):
        return 'yclass'
    def fake_X(self):
        return 'fakeX'
    def fake_y(self):
        return 'fakey'
    def get_pred(self, path):
        return 'pred'
    def save_pred(self, y_pred, path):
        return 'pred'
"""

    import_module('opener', script)

    o = load_from_module()

    # create some data folders
    data_root_path = o._workspace.data_folder
    data_paths = [os.path.join(data_root_path, str(i)) for i in range(5)]
    [os.makedirs(p) for p in data_paths]

    o.data_folders = data_paths
    assert o.get_X() == 'Xclass'
