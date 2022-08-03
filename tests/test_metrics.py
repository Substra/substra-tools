import json
import sys

import pytest
import numpy as np

from substratools import metrics
from substratools.utils import import_module
from substratools.workspace import MetricsWorkspace


@pytest.fixture()
def write_pred_file():
    workspace = MetricsWorkspace()
    data = list(range(3, 6))
    with open(workspace.output_predictions_path, "w") as f:
        json.dump(data, f)
    return workspace.output_predictions_path, data


@pytest.fixture
def load_float_metrics_module():
    code = """
from substratools import Metrics
class FloatMetrics(Metrics):
    def score(self, y_true, y_pred):
        return sum(y_true) + sum(y_pred)
"""
    import_module("metrics", code)
    yield
    del sys.modules["metrics"]


@pytest.fixture
def load_np_metrics_module():
    code = """
from substratools import Metrics
import numpy as np
class FloatNpMetrics(Metrics):
    def score(self, y_true, y_pred):
        return np.float32(0.99)
"""
    import_module("metrics", code)
    yield
    del sys.modules["metrics"]


@pytest.fixture
def load_int_metrics_module():
    code = """
from substratools import Metrics
class IntMetrics(Metrics):
    def score(self, y_true, y_pred):
        return int(1)
"""
    import_module("metrics", code)
    yield
    del sys.modules["metrics"]


@pytest.fixture
def load_dict_metrics_module():
    code = """
from substratools import Metrics
class DictMetrics(Metrics):
    def score(self, y_true, y_pred):
        return {"a": 1}
"""
    import_module("metrics", code)
    yield
    del sys.modules["metrics"]


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


def test_execute(load_float_metrics_module):
    s = metrics.execute(sysargs=[])
    assert s == 15


@pytest.mark.parametrize(
    "fake_data_mode,expected_score",
    [
        ([], 15),
        (["--fake-data", "--n-fake-samples", "3"], 0),
        (["--fake-data-mode", metrics.FakeDataMode.DISABLED.name, "--n-fake-samples", "3"], 15),
        (["--fake-data-mode", metrics.FakeDataMode.FAKE_Y.name, "--n-fake-samples", "3"], 12),
        (["--fake-data-mode", metrics.FakeDataMode.FAKE_Y_PRED.name, "--n-fake-samples", "3"], 0),
    ],
)
def test_execute_fake_data_modes(fake_data_mode, expected_score, load_float_metrics_module):
    s = metrics.execute(sysargs=fake_data_mode)
    assert s == expected_score


def test_execute_np(load_np_metrics_module):
    s = metrics.execute(sysargs=[])
    assert s == pytest.approx(0.99)


def test_execute_int(load_int_metrics_module):
    s = metrics.execute(sysargs=[])
    assert s == 1


def test_execute_dict(load_dict_metrics_module):
    # should raise an error
    with pytest.raises(TypeError):
        metrics.execute(sysargs=[])
