import json
import uuid
from os import PathLike
from typing import Any
from typing import TypedDict

import numpy as np
import pytest

from substratools import MetricAlgo
from substratools import algo
from substratools import load_performance
from substratools import opener
from substratools import save_performance
from substratools.task_resources import TaskResources
from substratools.workspace import AlgoWorkspace
from tests import utils
from tests.utils import InputIdentifiers
from tests.utils import OutputIdentifiers


@pytest.fixture()
def write_pred_file(workdir):
    pred_file = str(workdir / str(uuid.uuid4()))
    data = list(range(3, 6))
    with open(pred_file, "w") as f:
        json.dump(data, f)
    return pred_file, data


@pytest.fixture
def inputs(workdir, valid_opener_script, write_pred_file):
    return [
        {"id": InputIdentifiers.predictions, "value": str(write_pred_file[0]), "multiple": False},
        {"id": InputIdentifiers.datasamples, "value": str(workdir / "datasamples_unused"), "multiple": True},
        {"id": InputIdentifiers.opener, "value": str(valid_opener_script), "multiple": False},
    ]


@pytest.fixture
def outputs(workdir):
    return [{"id": OutputIdentifiers.performance, "value": str(workdir / str(uuid.uuid4())), "multiple": False}]


@pytest.fixture(autouse=True)
def setup(valid_opener, write_pred_file):
    pass


class FloatMetrics(MetricAlgo):
    def score(
        self,
        inputs: TypedDict("inputs", {InputIdentifiers.datasamples: Any, InputIdentifiers.predictions: Any}),
        outputs: TypedDict("outputs", {OutputIdentifiers.performance: PathLike}),
        task_properties: TypedDict("task_properties", {InputIdentifiers.rank: int}),
    ):
        y_true = inputs.get(InputIdentifiers.datasamples)[1]
        y_pred_path = inputs.get(InputIdentifiers.predictions)
        y_pred = utils.load_predictions(y_pred_path)

        score = sum(y_true) + sum(y_pred)

        save_performance(performance=score, path=outputs.get(OutputIdentifiers.performance))


def test_create():
    FloatMetrics()


def test_score(workdir, write_pred_file):
    inputs = TaskResources(
        json.dumps(
            [
                {"id": InputIdentifiers.predictions, "value": str(write_pred_file[0]), "multiple": False},
            ]
        )
    )
    outputs = TaskResources(
        json.dumps(
            [{"id": OutputIdentifiers.performance, "value": str(workdir / str(uuid.uuid4())), "multiple": False}]
        )
    )
    m = FloatMetrics()
    workspace = AlgoWorkspace(inputs=inputs, outputs=outputs)
    wp = algo.GenericAlgoWrapper(m, workspace=workspace, opener_wrapper=opener.load_from_module())
    wp.execute(method_name="score")
    s = load_performance(wp._workspace.task_outputs[OutputIdentifiers.performance])
    assert s == 15


def test_execute(inputs, outputs):
    perf_path = outputs[0]["value"]
    algo.execute(
        FloatMetrics(),
        sysargs=["--method-name", "score", "--inputs", json.dumps(inputs), "--outputs", json.dumps(outputs)],
    )
    s = load_performance(perf_path)
    assert s == 15


@pytest.mark.parametrize(
    "fake_data_mode,expected_score",
    [
        ([], 15),
        (["--fake-data", "--n-fake-samples", "3"], 12),
    ],
)
def test_execute_fake_data_modes(fake_data_mode, expected_score, inputs, outputs):
    perf_path = outputs[0]["value"]
    algo.execute(
        FloatMetrics(),
        sysargs=fake_data_mode
        + ["--method-name", "score", "--inputs", json.dumps(inputs), "--outputs", json.dumps(outputs)],
    )
    s = load_performance(perf_path)
    assert s == expected_score


def test_execute_np(inputs, outputs):
    class FloatNpMetrics(MetricAlgo):
        def score(
            self,
            inputs,
            outputs,
            task_properties: dict,
        ):
            save_performance(np.float64(0.99), outputs.get(OutputIdentifiers.performance))

    perf_path = outputs[0]["value"]
    algo.execute(
        FloatNpMetrics(),
        sysargs=["--method-name", "score", "--inputs", json.dumps(inputs), "--outputs", json.dumps(outputs)],
    )
    s = load_performance(perf_path)
    assert s == pytest.approx(0.99)


def test_execute_int(inputs, outputs):
    class IntMetrics(MetricAlgo):
        def score(
            self,
            inputs,
            outputs,
            task_properties: dict,
        ):
            save_performance(int(1), outputs.get(OutputIdentifiers.performance))

    perf_path = outputs[0]["value"]
    algo.execute(
        IntMetrics(),
        sysargs=["--method-name", "score", "--inputs", json.dumps(inputs), "--outputs", json.dumps(outputs)],
    )
    s = load_performance(perf_path)
    assert s == 1


def test_execute_dict(inputs, outputs):
    class DictMetrics(MetricAlgo):
        def score(
            self,
            inputs,
            outputs,
            task_properties: dict,
        ):
            save_performance({"a": 1}, outputs.get(OutputIdentifiers.performance))

    perf_path = outputs[0]["value"]
    algo.execute(
        DictMetrics(),
        sysargs=["--method-name", "score", "--inputs", json.dumps(inputs), "--outputs", json.dumps(outputs)],
    )
    s = load_performance(perf_path)
    assert s["a"] == 1
