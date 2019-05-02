import json
import sys

from substratools import metrics
from substratools.workspace import Workspace
from substratools.utils import import_module

import pytest


@pytest.fixture()
def write_pred_file():
    workspace = Workspace()
    data = list(range(3, 6))
    with open(workspace.pred_filepath, 'w') as f:
        json.dump(data, f)
    return workspace.pred_filepath, data


@pytest.fixture
def load_metrics_module():
    code = """
from substratools import Metrics
class DummyMetrics(Metrics):
    def score(self, y_true, y_pred):
        return sum(y_true) + sum(y_pred)
"""
    import_module('metrics', code)
    yield
    del sys.modules['metrics']


@pytest.fixture(autouse=True)
def setup(valid_opener, write_pred_file):
    pass


class DummyMetrics(metrics.Metrics):
    def score(self, y_true, y_pred):
        return sum(y_true) + sum(y_pred)


def test_create():
    DummyMetrics()


def test_score():
    m = DummyMetrics()
    wp = metrics.MetricsWrapper(m)
    s = wp.score()
    assert s == 15


def test_execute(load_metrics_module):
    s = metrics.execute()
    assert s == 15


def test_execute_dryrun(load_metrics_module):
    s = metrics.execute(dry_run=True)
    assert s == 0
