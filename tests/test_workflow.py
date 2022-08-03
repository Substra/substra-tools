import json
import os

import pytest

from substratools import Algo
from substratools import Metrics
from substratools.algo import AlgoWrapper
from substratools.metrics import MetricsWrapper
from substratools.utils import import_module
from substratools.workspace import AlgoWorkspace


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

    def get_predictions(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def save_predictions(self, pred, path):
        with open(path, 'w') as f:
            json.dump(pred, f)
"""
    import_module("opener", script)


class DummyAlgo(Algo):
    def train(self, X, y, models, rank):
        total = sum([m["i"] for m in models])
        new_model = {"i": len(models) + 1, "total": total}
        return new_model

    def predict(self, X, model):
        return {"sum": model["i"]}

    def load_model(self, path):
        with open(path, "r") as f:
            return json.load(f)

    def save_model(self, model, path):
        with open(path, "w") as f:
            json.dump(model, f)


class DummyMetrics(Metrics):
    def score(self, y, pred):
        return pred["sum"]


def test_workflow(workdir, dummy_opener):
    a = DummyAlgo()
    loop1_model_path = workdir / "loop1model"
    loop1_workspace = AlgoWorkspace(output_model_path=str(loop1_model_path))
    loop1_wp = AlgoWrapper(a, workspace=loop1_workspace)

    # loop 1 (no input)
    model = loop1_wp.train()
    assert model == {"i": 1, "total": 0}
    assert os.path.exists(loop1_model_path)

    loop2_model_path = workdir / "loop2model"
    loop2_workspace = AlgoWorkspace(input_model_paths=[str(loop1_model_path)], output_model_path=str(loop2_model_path))
    loop2_wp = AlgoWrapper(a, workspace=loop2_workspace)

    # loop 2 (one model as input)
    model = loop2_wp.train()
    assert model == {"i": 2, "total": 1}
    assert os.path.exists(loop2_model_path)

    loop3_model_path = workdir / "loop2model"
    loop3_workspace = AlgoWorkspace(
        input_model_paths=[str(loop1_model_path), str(loop2_model_path)], output_model_path=str(loop3_model_path)
    )
    loop3_wp = AlgoWrapper(a, workspace=loop3_workspace)

    # loop 3 (two models as input)
    model = loop3_wp.train()
    assert model == {"i": 3, "total": 3}
    assert os.path.exists(loop3_model_path)

    predict_workspace = AlgoWorkspace(
        input_model_paths=[str(loop3_model_path)],
    )
    predict_wp = AlgoWrapper(a, workspace=predict_workspace)

    # predict
    pred = predict_wp.predict()
    assert pred == {"sum": 3}

    # metrics
    metrics_wp = MetricsWrapper(DummyMetrics())
    score = metrics_wp.score()
    assert score == 3.0
