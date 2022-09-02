import json
import os

import pytest

from substratools import Algo
from substratools import Metrics
from substratools import load_performance
from substratools import opener
from substratools import save_performance
from substratools.algo import GenericAlgoWrapper
from substratools.metrics import MetricsWrapper
from substratools.task_resources import TaskResources
from substratools.utils import import_module
from substratools.workspace import AlgoWorkspace
from tests import utils


@pytest.fixture
def dummy_opener():
    script = """
import json
from substratools import Opener

class DummyOpener(Opener):
    def get_X(self, folder):
        return None

    def get_y(self, folder):
        return None

    def fake_X(self, n_samples):
        raise NotImplementedError

    def fake_y(self, n_samples):
        raise NotImplementedError
"""
    import_module("opener", script)


# TODO change algo
class DummyAlgo(Algo):
    def train(self, inputs, outputs):

        models = utils.load_models(inputs.get("models", []))
        total = sum([m["i"] for m in models])
        new_model = {"i": len(models) + 1, "total": total}

        utils.save_model(new_model, outputs.get("model"))

    def predict(self, inputs, outputs):
        model = utils.load_model(inputs.get("model"))
        pred = {"sum": model["i"]}
        utils.save_predictions(pred, outputs.get("predictions"))


class DummyMetrics(Metrics):
    def score(self, inputs, outputs):
        y_pred_path = inputs.get("predictions")
        y_pred = utils.load_predictions(y_pred_path)

        score = y_pred["sum"]

        save_performance(performance=score, path=outputs.get("performance"))


def test_workflow(workdir, dummy_opener):

    a = DummyAlgo()
    loop1_model_path = workdir / "loop1model"
    loop1_workspace_outputs = TaskResources(
        json.dumps([{"id": "model", "value": str(loop1_model_path), "multiple": False}])
    )
    loop1_workspace = AlgoWorkspace(outputs=loop1_workspace_outputs)
    loop1_wp = GenericAlgoWrapper(a, workspace=loop1_workspace, opener_wrapper=None)

    # loop 1 (no input)
    loop1_wp.task_launcher(method_name="train")
    model = utils.load_model(path=loop1_wp._workspace.task_outputs["model"])

    assert model == {"i": 1, "total": 0}
    assert os.path.exists(loop1_model_path)

    loop2_model_path = workdir / "loop2model"

    loop2_workspace_inputs = TaskResources(
        json.dumps([{"id": "model", "value": str(loop1_model_path), "multiple": False}])
    )
    loop2_workspace_outputs = TaskResources(
        json.dumps([{"id": "model", "value": str(loop2_model_path), "multiple": False}])
    )
    loop2_workspace = AlgoWorkspace(inputs=loop2_workspace_inputs, outputs=loop2_workspace_outputs)
    loop2_wp = GenericAlgoWrapper(a, workspace=loop2_workspace, opener_wrapper=None)

    # loop 2 (one model as input)
    loop2_wp.task_launcher(method_name="train")
    model = utils.load_model(path=loop2_wp._workspace.task_outputs["model"])
    assert model == {"i": 2, "total": 1}
    assert os.path.exists(loop2_model_path)

    loop3_model_path = workdir / "loop2model"
    loop3_workspace_inputs = TaskResources(
        json.dumps(
            [
                {"id": "model", "value": str(loop1_model_path), "multiple": False},
                {"id": "model", "value": str(loop2_model_path), "multiple": False},
            ]
        )
    )
    loop3_workspace_outputs = TaskResources(
        json.dumps([{"id": "model", "value": str(loop3_model_path), "multiple": False}])
    )
    loop3_workspace = AlgoWorkspace(inputs=loop3_workspace_inputs, outputs=loop3_workspace_outputs)
    loop3_wp = GenericAlgoWrapper(a, workspace=loop3_workspace, opener_wrapper=None)

    # loop 3 (two models as input)
    loop3_wp.task_launcher("train")
    model = utils.load_model(path=loop3_wp._workspace.output_model_path)
    assert model == {"i": 3, "total": 3}
    assert os.path.exists(loop3_model_path)

    predict_workspace = AlgoWorkspace(
        input_model_paths=[str(loop3_model_path)],
    )
    predict_wp = GenericAlgoWrapper(a, workspace=predict_workspace, opener_wrapper=opener.load_from_module())

    # predict
    predict_wp.predict()
    pred = utils.load_predictions(path=predict_wp._workspace.output_predictions_path)
    assert pred == {"sum": 3}

    # metrics
    metrics_wp = MetricsWrapper(DummyMetrics())
    metrics_wp.score()
    score = load_performance(path=metrics_wp._workspace.output_perf_path)
    assert score == 3.0
