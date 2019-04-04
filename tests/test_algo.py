import json
import os

from substratools import algo

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
    class OpenerInterface():
        def get_X(self):
            return 'X'

        def get_y(self):
            return 'y'

        def fake_X(self):
            return self.get_X() + 'fake'

        def fake_y(self):
            return self.get_y() + 'fake'

        def get_pred(self, folder):
            return 'pred'

        def save_pred(self, pred):
            return pred

    yield OpenerInterface()


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

        def predict(self, models):
            pass

    yield DummyAlgo


def test_algo_create(dummy_algo_class):
    # check we can instantiate a dummy algo class
    dummy_algo_class()


def test_algo_train_no_model(dummy_algo_class):
    a = dummy_algo_class()
    pred, model = a._execute_train(model_paths=[])
    assert pred == 'Xy'
    assert model == 1


def test_algo_train_multiple_models(dummy_algo_class, workdir):
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

    pred, model = a._execute_train(model_paths=model_filenames)
    assert pred == 'Xy'
    assert model == 3


def test_algo_train_dry_run(dummy_algo_class):
    a = dummy_algo_class()
    pred, model = a._execute_train(model_paths=[], dry_run=True)
    assert pred == 'Xfakeyfake'
    assert model == 1


def test_algo_execute(dummy_algo_class, workdir):
    a = dummy_algo_class()

    cli = algo._generate_cli(a)
    runner = CliRunner()
    result = runner.invoke(cli, ['train', '--dry-run'])
    print(result.exception)
    assert result.exit_code == 0

    result = runner.invoke(cli, ['train'])
    print(result.exception)
    assert result.exit_code == 0
