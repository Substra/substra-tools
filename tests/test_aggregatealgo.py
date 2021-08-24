import json
import types

from substratools import algo, exceptions

import pytest


@pytest.fixture(autouse=True)
def setup(valid_opener):
    pass


class DummyAggregateAlgo(algo.AggregateAlgo):

    def aggregate(self, models, rank):
        new_model = {'value': 0}
        for m in models:
            new_model['value'] += m['value']
        return new_model

    def predict(self, X, model):
        pred = model['value']
        return X * pred

    def load_model(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def save_model(self, model, path):
        with open(path, 'w') as f:
            json.dump(model, f)


class NoSavedModelAggregateAlgo(DummyAggregateAlgo):
    def save_model(self, model, path):
        # do not save model at all
        pass


class WrongSavedModelAggregateAlgo(DummyAggregateAlgo):
    def save_model(self, model, path):
        # simulate numpy.save behavior
        with open(path + '.npy', 'w') as f:
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


@pytest.mark.parametrize("fake_data,expected_pred,n_fake_samples", [
    (False, 'X', None),
    (True, ['Xfake'], 1),
])
def test_predict(fake_data, expected_pred, n_fake_samples, workdir, create_models):
    _, model_filenames = create_models

    a = DummyAggregateAlgo()
    wp = algo.AggregateAlgoWrapper(a)
    pred = wp.predict(model_filenames[0], fake_data=fake_data, n_fake_samples=n_fake_samples)
    assert pred == expected_pred


def test_execute_aggregate(workdir):

    output_model_path = workdir / 'model' / 'model'
    assert not output_model_path.exists()

    algo.execute(DummyAggregateAlgo(), sysargs=['aggregate'])
    assert output_model_path.exists()

    output_model_path.unlink()
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


def test_execute_predict(workdir, create_models):
    _, model_filenames = create_models
    model_name = 'model'
    output_model_path = workdir / 'model' / model_name
    assert not output_model_path.exists()

    command = ['aggregate']
    command.extend(model_filenames)
    algo.execute(DummyAggregateAlgo(), sysargs=command)
    assert output_model_path.exists()

    # do predict on output model
    pred_path = workdir / 'pred' / 'pred'
    assert not pred_path.exists()
    algo.execute(DummyAggregateAlgo(), sysargs=['predict', model_name])
    assert pred_path.exists()
    with open(pred_path, 'r') as f:
        pred = json.load(f)
    assert pred == 'XXX'
    pred_path.unlink()


@pytest.mark.parametrize('algo_class', (NoSavedModelAggregateAlgo, WrongSavedModelAggregateAlgo))
def test_model_check(algo_class):
    a = algo_class()
    wp = algo.AggregateAlgoWrapper(a)

    with pytest.raises(exceptions.MissingFileError):
        wp.aggregate([])


@pytest.mark.parametrize('use_models_generator,models_type', (
    (True, types.GeneratorType),
    (False, list),
))
def test_models_generator(mocker, workdir, create_models, use_models_generator, models_type):
    _, model_filenames = create_models

    command = ['aggregate']
    command.extend(model_filenames)

    a = DummyAggregateAlgo()
    a.use_models_generator = use_models_generator
    mocker.patch.object(a, 'aggregate', autospec=True, return_value={})

    algo.execute(a, sysargs=command)
    models = a.aggregate.call_args[0][0]
    assert isinstance(models, models_type)
