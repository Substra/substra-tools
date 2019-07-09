import json
import os
import shutil

from substratools import Algo, Metrics
from substratools.algo import AlgoWrapper
from substratools.metrics import MetricsWrapper
from substratools.utils import import_module

import pytest


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

    def fake_X(self):
        raise NotImplementedError

    def fake_y(self):
        raise NotImplementedError

    def get_pred(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def save_pred(self, pred, path):
        with open(path, 'w') as f:
            json.dump(pred, f)
"""
    import_module('opener', script)


class DummyAlgo(Algo):
    def train(self, X, y, models, rank):
        total = sum([m['i'] for m in models])
        pred = {'sum': len(models)}
        new_model = {'i': len(models) + 1, 'total': total}
        return pred, new_model

    def predict(self, X, model):
        return {'sum': model['i']}

    def load_model(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def save_model(self, model, path):
        with open(path, 'w') as f:
            json.dump(model, f)


class DummyMetrics(Metrics):
    def score(self, y, pred):
        return pred


def test_workflow(workdir, dummy_opener):
    algo_wp = AlgoWrapper(DummyAlgo())

    models_path = algo_wp._workspace.model_folder

    # loop 1 (no input)
    pred, model = algo_wp.train([])
    assert pred == {'sum': 0}
    assert model == {'i': 1, 'total': 0}
    output_model_path = os.path.join(models_path, 'model')
    assert os.path.exists(output_model_path)

    # loop 2 (one model as input)
    model_1_name = 'model1'
    shutil.move(output_model_path, os.path.join(models_path, model_1_name))
    pred, model = algo_wp.train([model_1_name])
    assert pred == {'sum': 1}
    assert model == {'i': 2, 'total': 1}
    output_model_path = os.path.join(models_path, 'model')
    assert os.path.exists(output_model_path)

    # loop 3 (two models as input)
    model_2_name = 'model2'
    shutil.move(output_model_path, os.path.join(models_path, model_2_name))
    pred, model = algo_wp.train([model_1_name, model_2_name])
    assert pred == {'sum': 2}
    assert model == {'i': 3, 'total': 3}
    output_model_path = os.path.join(models_path, 'model')
    assert os.path.exists(output_model_path)

    # predict
    model_3_name = 'model3'
    shutil.move(output_model_path, os.path.join(models_path, model_3_name))
    pred = algo_wp.predict(model_3_name)
    assert pred == {'sum': 3}

    # metrics
    metrics_wp = MetricsWrapper(DummyMetrics())
    score = metrics_wp.score()
    assert score == {'sum': 3}
