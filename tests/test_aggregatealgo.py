import json

from substratools import algo

import pytest


class DummyAggregateAlgo(algo.AggregateAlgo):

    def aggregate(self, models, rank):
        new_model = {'value': 0}
        for m in models:
            new_model['value'] += m['value']
        return new_model

    def load_model(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def save_model(self, model, path):
        with open(path, 'w') as f:
            json.dump(model, f)


@pytest.fixture
def create_models(workdir):
    model_a = {'value': 1}
    model_b = {'value': 2}

    model_dir = workdir / "model"
    model_dir.mkdir()

    def _create_model(model_data):
        model_name = model_data['value']
        filename = "{}.json".format(model_name)
        path = model_dir / filename
        path.write_text(json.dumps(model_data))
        return filename

    model_datas = [model_a, model_b]
    model_filenames = [_create_model(d) for d in model_datas]

    return model_datas, model_filenames


def test_create():
    # check we can instantiate a dummy algo class
    DummyAggregateAlgo()


def test_aggregate_no_model():
    a = DummyAggregateAlgo()
    wp = algo.AggregateAlgoWrapper(a)
    model = wp.aggregate([])
    assert model['value'] == 0


def test_aggregate_multiple_models(workdir, create_models):
    _, model_filenames = create_models

    a = DummyAggregateAlgo()
    wp = algo.AggregateAlgoWrapper(a)

    model = wp.aggregate(model_filenames)
    assert model['value'] == 3


def test_execute_aggregate(workdir):

    output_model_path = workdir / 'model' / 'model'
    assert not output_model_path.exists()

    algo.execute(DummyAggregateAlgo(), sysargs=['aggregate'])
    assert output_model_path.exists()

    algo.execute(DummyAggregateAlgo(), sysargs=['aggregate', '--debug'])
    assert output_model_path.exists()


def test_execute_aggregate_multiple_models(workdir, create_models):
    _, model_filenames = create_models

    output_model_path = workdir / 'model' / 'model'
    assert not output_model_path.exists()

    command = ['aggregate']
    command.extend(model_filenames)

    algo.execute(DummyAggregateAlgo(), sysargs=command)
    assert output_model_path.exists()
    with open(output_model_path, 'r') as f:
        model = json.load(f)
    assert model['value'] == 3
