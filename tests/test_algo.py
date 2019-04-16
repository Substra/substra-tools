import json

from substratools import algo, Opener

import pytest


@pytest.fixture
def dummy_opener():
    # fake opener module using a class
    class FakeOpener(Opener):
        def get_X(self, folder):
            return 'X'

        def get_y(self, folder):
            return 'y'

        def fake_X(self):
            return 'Xfake'

        def fake_y(self):
            return 'yfake'

        def get_pred(self, path):
            with open(path, 'r') as f:
                return json.load(f)

        def save_pred(self, pred, path):
            with open(path, 'w') as f:
                json.dump(pred, f)

    yield FakeOpener()


@pytest.fixture
def dummy_algo_class(dummy_opener):
    class DummyAlgo(algo.Algo):
        OPENER = dummy_opener

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

    yield DummyAlgo


def test_create(dummy_algo_class):
    # check we can instantiate a dummy algo class
    dummy_algo_class()


def test_train_no_model(dummy_algo_class):
    a = dummy_algo_class()
    wp = algo.AlgoWrapper(a)
    pred, model = wp.train([])
    assert pred == 'Xy'
    assert model == 1


def test_train_multiple_models(dummy_algo_class, workdir):
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

    a = dummy_algo_class()
    wp = algo.AlgoWrapper(a)

    pred, model = wp.train(model_filenames)
    assert pred == 'Xy'
    assert model == 3


def test_train_dry_run(dummy_algo_class):
    a = dummy_algo_class()
    wp = algo.AlgoWrapper(a)
    pred, model = wp.train([], dry_run=True)
    assert pred == 'Xfakeyfake'
    assert model == 1


def test_predict(dummy_algo_class):
    a = dummy_algo_class()
    wp = algo.AlgoWrapper(a)
    pred = wp.predict([])
    assert pred == ''


def test_execute_train(dummy_algo_class, workdir):

    output_model_path = workdir / 'model' / 'model'
    assert not output_model_path.exists()

    algo.execute(dummy_algo_class(), sysargs=['train'])
    assert output_model_path.exists()

    algo.execute(dummy_algo_class(), sysargs=['train', '--dry-run'])


def test_execute_predict(dummy_algo_class, workdir):
    pred_path = workdir / 'pred' / 'pred'
    assert not pred_path.exists()
    algo.execute(dummy_algo_class(), sysargs=['predict'])
    assert pred_path.exists()
