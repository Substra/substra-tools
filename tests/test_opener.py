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


@pytest.mark.skip(reason="not supported")
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
def get_predictions(path):
    return 'pred'
def save_predictions(y_pred, path):
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
    def get_predictions(self, path):
        return 'pred'
    def save_predictions(self, y_pred, path):
        return 'pred'
"""

    import_module('opener', script)

    o = load_from_module()
    assert o.get_X() == 'Xclass'


def test_load_opener_from_path(tmp_cwd, valid_opener_code):
    dirpath = tmp_cwd / 'myopener'
    dirpath.mkdir()
    path = dirpath / 'my_opener.py'
    path.write_text(valid_opener_code)
    o = load_from_module(path=path)
    assert o.get_X() == 'X'


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
    def get_predictions(self, path):
        return 'pred'
    def save_predictions(self, y_pred, path):
        return 'pred'
"""

    import_module('opener', script)

    o = load_from_module()

    # create some data folders
    data_root_path = o._workspace.input_data_folder_path
    data_paths = [os.path.join(data_root_path, str(i)) for i in range(5)]
    [os.makedirs(p) for p in data_paths]

    o.data_folders = data_paths
    assert o.get_X() == 'Xclass'


@pytest.mark.parametrize('save_predictions_method_body', (
    """
        pass
    """,
    """
        with open(path + '.npy', 'w') as f:
            json.dump(pred, f)
    """,
))
def test_predictions_check(tmp_cwd, save_predictions_method_body):
    script = f"""
import json
from substratools import Opener

class MyOpener(Opener):
    def get_X(self, folder):
        return 'X'

    def get_y(self, folder):
        return list(range(0, 3))

    def fake_X(self):
        return 'Xfake'

    def fake_y(self):
        return [0] * 3

    def get_predictions(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def save_predictions(self, pred, path):
        {save_predictions_method_body}
"""
    import_module('opener', script)

    o = load_from_module()

    with pytest.raises(AssertionError):
        o.save_predictions({'foo': 'bar'})
