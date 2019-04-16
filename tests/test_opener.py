import imp
import os
import sys

import pytest

from substratools import exceptions
from substratools.opener import load_from_module, OpenerWrapper


@pytest.fixture
def tmp_cwd(tmp_path):
    # create a temporary current working directory
    new_dir = tmp_path / "workspace"
    new_dir.mkdir()

    old_dir = os.getcwd()
    os.chdir(new_dir)

    yield new_dir

    os.chdir(old_dir)


def import_opener(code):
    module = imp.new_module('opener')
    sys.modules['opener'] = module
    exec(code, module.__dict__)


def test_load_opener_not_found(tmp_cwd):
    with pytest.raises(ModuleNotFoundError):
        load_from_module()


def test_load_invalid_opener(tmp_cwd):
    invalid_script = """
def get_X():
    raise NotImplementedError
def get_y():
    raise NotImplementedError
"""

    import_opener(invalid_script)

    with pytest.raises(exceptions.InvalidOpener):
        load_from_module()


def test_load_opener_as_module(tmp_cwd):
    script = """
def _helper():
    pass
def get_X(folder):
    return 'X'
def get_y(folder):
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

    import_opener(script)

    o = load_from_module()
    assert o.get_X() == 'X'


def test_load_opener_as_class(tmp_cwd):
    script = """
from substratools import Opener
class MyOpener(Opener):
    def get_X(self, folder):
        return 'Xclass'
    def get_y(self, folder):
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

    import_opener(script)

    o = load_from_module()
    assert o.get_X() == 'Xclass'


def test_opener_wrapper_invalid_interface():
    with pytest.raises(exceptions.InvalidOpener):
        OpenerWrapper(opener='invalid')
