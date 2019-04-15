import json
import os

from substratools import algo, Opener

import pytest
from click.testing import CliRunner


@pytest.fixture
def workdir(tmp_path):
    d = tmp_path / "substra-workspace"
    d.mkdir()
    return d


@pytest.fixture(autouse=True)
def patch_cwd(monkeypatch, workdir):
    # this is needed to ensure the workspace is located in a tmpdir
    def getcwd():
        return str(workdir)
    monkeypatch.setattr(os, 'getcwd', getcwd)


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
            return 'pred'

        def save_pred(self, pred, path):
            return pred

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
    pred, model = wp.train(model_paths=[])
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

    pred, model = wp.train(model_paths=model_filenames)
    assert pred == 'Xy'
    assert model == 3


def test_train_dry_run(dummy_algo_class):
    a = dummy_algo_class()
    wp = algo.AlgoWrapper(a)
    pred, model = wp.train(model_paths=[], dry_run=True)
    assert pred == 'Xfakeyfake'
    assert model == 1


def test_predict(dummy_algo_class):
    a = dummy_algo_class()
    wp = algo.AlgoWrapper(a)
    pred = wp.predict(model_paths=[])
    assert pred == ''


def test_execute_train(dummy_algo_class, workdir):
    cli = algo._generate_cli(dummy_algo_class)
    runner = CliRunner()

    output_model_path = workdir / 'model' / 'model'

    assert not output_model_path.exists()

    result = runner.invoke(cli, ['train'])
    assert result.exit_code == 0
    assert output_model_path.exists()

    result = runner.invoke(cli, ['train', '--dry-run'])
    assert result.exit_code == 0


def test_execute_predict(dummy_algo_class, workdir):
    cli = algo._generate_cli(dummy_algo_class)
    runner = CliRunner()

    result = runner.invoke(cli, ['predict'])
    print(result.exception)
    assert result.exit_code == 0
