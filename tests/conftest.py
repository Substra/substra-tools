import os
import sys

import pytest

from substratools.utils import import_module


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

    monkeypatch.setattr(os, "getcwd", getcwd)


@pytest.fixture()
def valid_opener_code():
    return """
import json
from substratools import Opener

class FakeOpener(Opener):
    def get_X(self, folder):
        return 'X'

    def get_y(self, folder):
        return list(range(0, 3))

    def fake_X(self, n_samples):
        return ['Xfake'] * n_samples

    def fake_y(self, n_samples):
        return [0] * n_samples

    def get_predictions(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def save_predictions(self, pred, path):
        with open(path, 'w') as f:
            json.dump(pred, f)
"""


@pytest.fixture()
def valid_opener(valid_opener_code):
    import_module("opener", valid_opener_code)
    yield
    del sys.modules["opener"]
