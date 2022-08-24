import os

import pytest

from substratools import Algo
from substratools import Metrics
from substratools import load_performance
from substratools import save_performance
from substratools.algo import AlgoWrapper
from substratools.algo import InputIdentifiers
from substratools.algo import OutputIdentifiers
from substratools.metrics import MetricsWrapper
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

        models = utils.load_models(inputs.get(InputIdentifiers.models, []))
        total = sum([m["i"] for m in models])
        new_model = {"i": len(models) + 1, "total": total}

        utils.save_model(new_model, outputs.get(OutputIdentifiers.model))

    def predict(self, inputs, outputs):
        model = utils.load_model(inputs.get(InputIdentifiers.model))
        pred = {"sum": model["i"]}
        utils.save_predictions(pred, outputs.get(OutputIdentifiers.predictions))


class DummyMetrics(Metrics):
    def score(self, inputs, outputs):
        y_pred_path = inputs.get(InputIdentifiers.predictions)
        y_pred = utils.load_predictions(y_pred_path)

        score = y_pred["sum"]

        save_performance(performance=score, path=outputs.get(OutputIdentifiers.performance))


def test_workflow(workdir, dummy_opener):
    a = DummyAlgo()
    loop1_model_path = workdir / "loop1model"
    loop1_workspace = AlgoWorkspace(output_model_path=str(loop1_model_path))
    loop1_wp = AlgoWrapper(a, workspace=loop1_workspace)

    # loop 1 (no input)
    loop1_wp.train()
    model = utils.load_model(path=loop1_wp._workspace.output_model_path)

    assert model == {"i": 1, "total": 0}
    assert os.path.exists(loop1_model_path)

    loop2_model_path = workdir / "loop2model"
    loop2_workspace = AlgoWorkspace(input_model_paths=[str(loop1_model_path)], output_model_path=str(loop2_model_path))
    loop2_wp = AlgoWrapper(a, workspace=loop2_workspace)

    # loop 2 (one model as input)
    loop2_wp.train()
    model = utils.load_model(path=loop2_wp._workspace.output_model_path)
    assert model == {"i": 2, "total": 1}
    assert os.path.exists(loop2_model_path)

    loop3_model_path = workdir / "loop2model"
    loop3_workspace = AlgoWorkspace(
        input_model_paths=[str(loop1_model_path), str(loop2_model_path)], output_model_path=str(loop3_model_path)
    )
    loop3_wp = AlgoWrapper(a, workspace=loop3_workspace)

    # loop 3 (two models as input)
    loop3_wp.train()
    model = utils.load_model(path=loop3_wp._workspace.output_model_path)
    assert model == {"i": 3, "total": 3}
    assert os.path.exists(loop3_model_path)

    predict_workspace = AlgoWorkspace(
        input_model_paths=[str(loop3_model_path)],
    )
    predict_wp = AlgoWrapper(a, workspace=predict_workspace)

    # predict
    predict_wp.predict()
    pred = utils.load_predictions(path=predict_wp._workspace.output_predictions_path)
    assert pred == {"sum": 3}

    # metrics
    metrics_wp = MetricsWrapper(DummyMetrics())
    metrics_wp.score()
    score = load_performance(path=metrics_wp._workspace.output_perf_path)
    assert score == 3.0
