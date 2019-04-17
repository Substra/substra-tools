import os
import sys

from substratools.utils import import_module

import pytest


@pytest.fixture
def workdir(tmp_path):
    d = tmp_path / "substra-workspace"
    d.mkdir()
    return d


@pytest.fixture(autouse=True)
def patch_cwd(monkeypatch, workdir):
    # this is needed to ensure the workspace is located in a tmpdir
    def getcwd():
        return str(workdir)
    monkeypatch.setattr(os, 'getcwd', getcwd)


@pytest.fixture()
def valid_opener():
    script = """
import json
from substratools import Opener

class FakeOpener(Opener):
    def get_X(self, folder):
        return 'X'

    def get_y(self, folder):
        return 'y'

    def fake_X(self):
        return 'Xfake'

    def fake_y(self):
        return 'yfake'

    def get_pred(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def save_pred(self, pred, path):
        with open(path, 'w') as f:
            json.dump(pred, f)
"""
    import_module('opener', script)
    yield
    del sys.modules['opener']
