import json
import os
import shutil
import sys
from pathlib import Path

import pytest

from substratools.task_resources import TaskResources
from substratools.utils import import_module
from substratools.workspace import AlgoWorkspace
from substratools.workspace import CompositeAlgoWorkspace


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
"""


@pytest.fixture()
def valid_opener(valid_opener_code):
    import_module("opener", valid_opener_code)
    yield
    del sys.modules["opener"]


@pytest.fixture()
def output_model_path(workdir: Path) -> str:
    return workdir / "model" / "model"


@pytest.fixture()
def valid_algo_workspace(output_model_path: str) -> AlgoWorkspace:

    workspace_outputs = TaskResources(json.dumps([{"id": "model", "value": str(output_model_path), "multiple": False}]))

    workspace = AlgoWorkspace(outputs=workspace_outputs)

    return workspace


@pytest.fixture()
def valid_composite_algo_workspace(workdir) -> CompositeAlgoWorkspace:
    return CompositeAlgoWorkspace(
        output_head_model_path=str(workdir / "model" / "model_head"),
        output_trunk_model_path=str(workdir / "model" / "model_trunk"),
    )


@pytest.fixture(scope="session")
def session_dir():
    temp_dir = Path.cwd() / "local-assets-tools"
    temp_dir.mkdir(parents=True, exist_ok=True)

    yield temp_dir
    shutil.rmtree(temp_dir)
