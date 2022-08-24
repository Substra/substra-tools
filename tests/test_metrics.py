import json
import sys
from os import PathLike
from typing import Any
from typing import TypedDict

import pytest

from substratools import load_performance
from substratools import metrics
from substratools import save_performance
from substratools.algo import InputIdentifiers
from substratools.algo import OutputIdentifiers
from substratools.utils import import_module
from substratools.workspace import MetricsWorkspace
from tests import utils


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
from substratools import save_performance
from substratools.algo import InputIdentifiers
from substratools.algo import OutputIdentifiers
from tests import utils

class FloatMetrics(Metrics):
    def score(self, inputs, outputs):
        y_true = inputs.get(InputIdentifiers.y)
        y_pred_path = inputs.get(InputIdentifiers.predictions)
        y_pred = utils.load_predictions(y_pred_path)
        s = sum(y_true) + sum(y_pred)
        save_performance(s, outputs.get(OutputIdentifiers.performance))
"""
    import_module("metrics", code)
    yield
    del sys.modules["metrics"]


@pytest.fixture
def load_np_metrics_module():
    code = """
from substratools import Metrics
import numpy as np
from substratools import save_performance
from substratools.algo import OutputIdentifiers

class FloatNpMetrics(Metrics):
    def score(self, inputs, outputs):
        save_performance(np.float64(0.99), outputs.get(OutputIdentifiers.performance))
"""
    import_module("metrics", code)
    yield
    del sys.modules["metrics"]


@pytest.fixture
def load_int_metrics_module():
    code = """
from substratools import Metrics
from substratools import save_performance
from substratools.algo import OutputIdentifiers

class IntMetrics(Metrics):
    def score(self, inputs, outputs):
        save_performance(int(1), outputs.get(OutputIdentifiers.performance))
"""
    import_module("metrics", code)
    yield
    del sys.modules["metrics"]


@pytest.fixture
def load_dict_metrics_module():
    code = """
from substratools import Metrics
from substratools import save_performance
from substratools.algo import OutputIdentifiers

class DictMetrics(Metrics):
    def score(self, inputs, outputs):
        save_performance({"a": 1}, outputs.get(OutputIdentifiers.performance))
"""
    import_module("metrics", code)
    yield
    del sys.modules["metrics"]


@pytest.fixture(autouse=True)
def setup(valid_opener, write_pred_file):
    pass


class DummyMetrics(metrics.Metrics):
    def score(
        self,
        inputs: TypedDict("inputs", {InputIdentifiers.y: Any, InputIdentifiers.predictions: Any}),
        outputs: TypedDict("outputs", {OutputIdentifiers.performance: PathLike}),
    ):
        y_true = inputs.get(InputIdentifiers.y)
        y_pred_path = inputs.get(InputIdentifiers.predictions)
        y_pred = utils.load_predictions(y_pred_path)

        score = sum(y_true) + sum(y_pred)

        save_performance(performance=score, path=outputs.get(OutputIdentifiers.performance))


def test_create():
    DummyMetrics()


def test_score():
    m = DummyMetrics()
    wp = metrics.MetricsWrapper(m)
    wp.score()
    s = load_performance(wp._workspace.output_perf_path)
    assert s == 15


def test_execute(load_float_metrics_module):
    perf_path = metrics.execute(sysargs=[])
    s = load_performance(perf_path)
    assert s == 15


@pytest.mark.parametrize(
    "fake_data_mode,expected_score",
    [
        ([], 15),
        (["--fake-data", "--n-fake-samples", "3"], 12),
    ],
)
def test_execute_fake_data_modes(fake_data_mode, expected_score, load_float_metrics_module):
    perf_path = metrics.execute(sysargs=fake_data_mode)
    s = load_performance(perf_path)
    assert s == expected_score


def test_execute_np(load_np_metrics_module):
    perf_path = metrics.execute(sysargs=[])
    s = load_performance(perf_path)
    assert s == pytest.approx(0.99)


def test_execute_int(load_int_metrics_module):
    perf_path = metrics.execute(sysargs=[])
    s = load_performance(perf_path)
    assert s == 1


def test_execute_dict(load_dict_metrics_module):
    perf_path = metrics.execute(sysargs=[])
    s = load_performance(perf_path)
    assert s["a"] == 1
