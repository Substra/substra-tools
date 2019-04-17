import json

from substratools import metrics
from substratools.workspace import Workspace

import pytest


@pytest.fixture()
def write_pred_file():
    workspace = Workspace()
    data = {'key': 'pred'}
    with open(workspace.pred_filepath, 'w') as f:
        json.dump(data, f)
    return workspace.pred_filepath, data


@pytest.fixture(autouse=True)
def setup(valid_opener, write_pred_file):
    pass


class DummyMetrics(metrics.Metrics):
    def score(self, y_true, y_pred):
        return y_true + y_pred['key']


def test_create():
    DummyMetrics()


def test_score():
    m = DummyMetrics()
    wp = metrics.MetricsWrapper(m)
    s = wp.score()
    assert s == 'ypred'


def test_execute():
    s = metrics._execute(DummyMetrics())
    assert s == 'ypred'
