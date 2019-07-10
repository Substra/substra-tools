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
    with open(workspace.output_predictions_path, 'w') as f:
        json.dump(data, f)
    return workspace.output_predictions_path, data


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


@pytest.mark.parametrize("dry_run_mode,expected_score", [
    ([], 15),
    (['--dry-run'], 0),
    (['--dry-run-mode', metrics.DryRunMode.DISABLED.name], 15),
    (['--dry-run-mode', metrics.DryRunMode.FAKE_Y.name], 12),
    (['--dry-run-mode', metrics.DryRunMode.FAKE_Y_PRED.name], 0),
])
def test_execute_dryrun_modes(dry_run_mode, expected_score,
                              load_metrics_module):
    s = metrics.execute(sysargs=dry_run_mode)
    assert s == expected_score
