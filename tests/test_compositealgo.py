import json
import pathlib

from substratools import algo
from substratools.workspace import CompositeAlgoWorkspace

import pytest


@pytest.fixture(autouse=True)
def setup(valid_opener):
    pass


class DummyCompositeAlgo(algo.CompositeAlgo):

    def train(self, X, y, head_model, trunk_model, rank):
        # init phase
        if head_model and trunk_model:
            new_head_model = dict(head_model)
            new_trunk_model = dict(trunk_model)
        else:
            new_head_model = {'value': 0}
            new_trunk_model = {'value': 0}

        # train models
        new_head_model['value'] += 1
        new_trunk_model['value'] -= 1

        # get predictions
        pred = list(range(new_head_model['value'], new_trunk_model['value']))

        return pred, new_head_model, new_trunk_model

    def predict(self, X, head_model, trunk_model):
        pred = list(range(head_model['value'], trunk_model['value']))
        return pred

    def load_model(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def save_model(self, model, path):
        with open(path, 'w') as f:
            json.dump(model, f)


@pytest.fixture
def workspace(workdir):
    models_dir = workdir / "input_models"
    models_dir.mkdir()
    return CompositeAlgoWorkspace(
        input_models_folder_path=str(models_dir),
    )


@pytest.fixture
def dummy_wrapper(workspace):
    a = DummyCompositeAlgo()
    return algo.CompositeAlgoWrapper(a, workspace=workspace)


@pytest.fixture
def create_models(workspace):
    head_model = {'value': 1}
    trunk_model = {'value': -1}

    def _create_model(model_data, name):
        filename = "{}.json".format(name)
        path = pathlib.Path(workspace.input_models_folder_path) / filename
        path.write_text(json.dumps(model_data))
        return filename

    return (
        [head_model, trunk_model],
        workspace.input_models_folder_path,
        _create_model(head_model,  'head'),
        _create_model(trunk_model, 'trunk')
    )


def test_create():
    # check we can instantiate a dummy algo class
    DummyCompositeAlgo()


def test_train_no_model(dummy_wrapper):
    pred, head_model, trunk_model = dummy_wrapper.train()
    assert pred == []
    assert head_model['value'] == 1
    assert trunk_model['value'] == -1


def test_train_input_head_trunk_models(create_models, dummy_wrapper):
    _, _, head_filename, trunk_filename = create_models

    pred, head_model, trunk_model = dummy_wrapper.train(head_filename, trunk_filename)
    assert pred == []
    assert head_model['value'] == 2
    assert trunk_model['value'] == -2


def test_train_fake_data(dummy_wrapper):
    pred, head_model, trunk_model = dummy_wrapper.train(fake_data=True)
    assert pred == []
    assert head_model['value'] == 1
    assert trunk_model['value'] == -1


@pytest.mark.parametrize("fake_data,expected_pred", [
    (False, []),
    (True, []),
])
def test_predict(fake_data, expected_pred, create_models, dummy_wrapper):
    _, _, head_filename, trunk_filename = create_models

    pred = dummy_wrapper.predict(head_filename, trunk_filename, fake_data=fake_data)
    assert pred == expected_pred


def test_execute_train(workdir):
    output_models_path = workdir / 'output_models'
    output_head_model_filename = 'head_model'
    output_trunk_model_filename = 'trunk_model'

    output_head_model_path = output_models_path / output_head_model_filename
    assert not output_head_model_path.exists()
    output_trunk_model_path = output_models_path / output_trunk_model_filename
    assert not output_trunk_model_path.exists()

    common_args = [
        '--output-models-path', str(output_models_path),
        '--output-head-model-filename', output_head_model_filename,
        '--output-trunk-model-filename', output_trunk_model_filename,
    ]

    algo.execute(DummyCompositeAlgo(), sysargs=['train'] + common_args)
    assert output_head_model_path.exists()
    assert output_trunk_model_path.exists()


def test_execute_train_multiple_models(workdir, create_models):
    _, input_models_folder, head_filename, trunk_filename = create_models

    output_models_folder_path = workdir / 'output_models'

    output_head_model_filename = 'output_head_model'
    output_head_model_path = output_models_folder_path / output_head_model_filename
    assert not output_head_model_path.exists()

    output_trunk_model_filename = 'output_trunk_model'
    output_trunk_model_path = output_models_folder_path / output_trunk_model_filename
    assert not output_trunk_model_path.exists()

    pred_path = workdir / 'pred' / 'pred'
    assert not pred_path.exists()

    command = [
        'train',
        '--input-models-path', str(input_models_folder),
        '--input-head-model-filename', head_filename,
        '--input-trunk-model-filename', trunk_filename,
        '--output-models-path', str(output_models_folder_path),
        '--output-head-model-filename', output_head_model_filename,
        '--output-trunk-model-filename', output_trunk_model_filename,
    ]

    algo.execute(DummyCompositeAlgo(), sysargs=command)
    assert output_head_model_path.exists()
    with open(output_head_model_path, 'r') as f:
        head_model = json.load(f)
    assert head_model['value'] == 2

    assert output_trunk_model_path.exists()
    with open(output_trunk_model_path, 'r') as f:
        trunk_model = json.load(f)
    assert trunk_model['value'] == -2

    assert pred_path.exists()
    with open(pred_path, 'r') as f:
        pred = json.load(f)
    assert pred == []


def test_execute_predict(workdir, create_models):
    _, input_models_folder, head_filename, trunk_filename = create_models

    pred_path = workdir / 'pred' / 'pred'
    assert not pred_path.exists()

    command = [
        'predict',
        '--input-models-path', str(input_models_folder),
        '--input-head-model-filename', head_filename,
        '--input-trunk-model-filename', trunk_filename,
    ]

    algo.execute(DummyCompositeAlgo(), sysargs=command)
    assert pred_path.exists()
    with open(pred_path, 'r') as f:
        pred = json.load(f)
    assert pred == []
    pred_path.unlink()
