import json

from substratools import algo

import pytest


@pytest.fixture(autouse=True)
def setup(valid_opener):
    pass


class DummyAlgo(algo.Algo):

    def train(self, X, y, models, rank):
        for m in models:
            assert isinstance(m, dict)
            assert 'name' in m

        pred = X + y
        model = len(models) + 1
        return pred, model

    def predict(self, X, y, models):
        pred = ''.join([m['name'] for m in models])
        return pred


@pytest.fixture
def create_models(workdir):
    model_a = {'name': 'a'}
    model_b = {'name': 'b'}

    model_dir = workdir / "model"
    model_dir.mkdir()

    def _create_model(model_data):
        model_name = model_data['name']
        filename = f"{model_name}.json"
        path = model_dir / filename
        path.write_text(json.dumps(model_data))
        return filename

    model_datas = [model_a, model_b]
    model_filenames = [_create_model(d) for d in model_datas]

    return model_datas, model_filenames


def test_create():
    # check we can instantiate a dummy algo class
    DummyAlgo()


def test_train_no_model():
    a = DummyAlgo()
    wp = algo.AlgoWrapper(a)
    pred, model = wp.train([])
    assert pred == 'Xy'
    assert model == 1


def test_train_multiple_models(workdir, create_models):
    _, model_filenames = create_models

    a = DummyAlgo()
    wp = algo.AlgoWrapper(a)

    pred, model = wp.train(model_filenames)
    assert pred == 'Xy'
    assert model == 3


def test_train_dry_run():
    a = DummyAlgo()
    wp = algo.AlgoWrapper(a)
    pred, model = wp.train([], dry_run=True)
    assert pred == 'Xfakeyfake'
    assert model == 1


def test_predict():
    a = DummyAlgo()
    wp = algo.AlgoWrapper(a)
    pred = wp.predict([])
    assert pred == ''


def test_execute_train(workdir):

    output_model_path = workdir / 'model' / 'model'
    assert not output_model_path.exists()

    algo.execute(DummyAlgo(), sysargs=['train'])
    assert output_model_path.exists()

    algo.execute(DummyAlgo(), sysargs=['train', '--dry-run'])


def test_execute_train_multiple_models(workdir, create_models):
    _, model_filenames = create_models

    output_model_path = workdir / 'model' / 'model'
    assert not output_model_path.exists()
    pred_path = workdir / 'pred' / 'pred'
    assert not pred_path.exists()

    command = ['train']
    command.extend(model_filenames)

    algo.execute(DummyAlgo(), sysargs=command)
    assert output_model_path.exists()
    with open(output_model_path, 'r') as f:
        model = json.load(f)
    assert model == 3

    assert pred_path.exists()
    with open(pred_path, 'r') as f:
        pred = json.load(f)
    assert pred == 'Xy'


def test_execute_predict(workdir):
    pred_path = workdir / 'pred' / 'pred'
    assert not pred_path.exists()
    algo.execute(DummyAlgo(), sysargs=['predict'])
    assert pred_path.exists()
