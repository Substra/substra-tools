from substratools import algo

import pytest
from click.testing import CliRunner


@pytest.fixture
def dummy_algo_class():
    class DummyAlgo(algo.Algo):
        def train(self, models, rank):
            pass

        def predict(self, models):
            pass

        def dry_run(self, models):
            pass

    yield DummyAlgo


def test_algo_create(dummy_algo_class):
    # check we can instantiate a dummy algo class
    dummy_algo_class()


def test_algo_execute(dummy_algo_class):
    a = dummy_algo_class()

    cli = algo._generate_cli(a)
    runner = CliRunner()
    result = runner.invoke(cli, ['dry-run', 'model_path'])
    assert result.exit_code == 0
