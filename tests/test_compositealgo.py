import json

import pytest

from substratools import algo
from substratools import exceptions
from substratools.task_resources import COMPOSITE_IO_LOCAL
from substratools.task_resources import COMPOSITE_IO_SHARED
from substratools.task_resources import TASK_IO_DATASAMPLES
from substratools.workspace import CompositeAlgoWorkspace


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
            new_head_model = {"value": 0}
            new_trunk_model = {"value": 0}

        # train models
        new_head_model["value"] += 1
        new_trunk_model["value"] -= 1

        return new_head_model, new_trunk_model

    def predict(self, X, head_model, trunk_model):
        pred = list(range(head_model["value"], trunk_model["value"]))
        return pred

    def load_head_model(self, path):
        return self._load_model(path)

    def save_head_model(self, model, path):
        return self._save_model(model, path)

    def load_trunk_model(self, path):
        return self._load_model(path)

    def save_trunk_model(self, model, path):
        return self._save_model(model, path)

    def _load_model(self, path):
        with open(path, "r") as f:
            return json.load(f)

    def _save_model(self, model, path):
        with open(path, "w") as f:
            json.dump(model, f)


class NoSavedTrunkModelAggregateAlgo(DummyCompositeAlgo):
    def save_trunk_model(self, model, path):
        # do not save model at all
        pass


class NoSavedHeadModelAggregateAlgo(DummyCompositeAlgo):
    def save_head_model(self, model, path):
        # do not save model at all
        pass


class WrongSavedTrunkModelAggregateAlgo(DummyCompositeAlgo):
    def save_trunk_model(self, model, path):
        # simulate numpy.save behavior
        with open(path + ".npy", "w") as f:
            json.dump(model, f)


class WrongSavedHeadModelAggregateAlgo(DummyCompositeAlgo):
    def save_head_model(self, model, path):
        # simulate numpy.save behavior
        with open(path + ".npy", "w") as f:
            json.dump(model, f)


@pytest.fixture
def workspace(workdir):
    output_dir = workdir / "outputs"
    output_dir.mkdir()
    return CompositeAlgoWorkspace(
        output_head_model_path=str(output_dir / "head"),
        output_trunk_model_path=str(output_dir / "trunk"),
        output_predictions_path=str(output_dir / "predictions"),
    )


@pytest.fixture
def dummy_wrapper(workspace):
    a = DummyCompositeAlgo()
    return algo.CompositeAlgoWrapper(a, workspace)


@pytest.fixture
def create_models(workdir):
    head_model = {"value": 1}
    trunk_model = {"value": -1}

    def _create_model(model_data, name):
        filename = "{}.json".format(name)
        path = workdir / filename
        path.write_text(json.dumps(model_data))
        return path

    return (
        [head_model, trunk_model],
        _create_model(head_model, "head"),
        _create_model(trunk_model, "trunk"),
    )


def test_create():
    # check we can instantiate a dummy algo class
    DummyCompositeAlgo()


def test_train_no_model(dummy_wrapper):
    head_model, trunk_model = dummy_wrapper.train()
    assert head_model["value"] == 1
    assert trunk_model["value"] == -1


def test_train_input_head_trunk_models(create_models, dummy_wrapper):
    _, head_path, trunk_path = create_models

    dummy_wrapper._workspace.input_head_model_path = head_path
    dummy_wrapper._workspace.input_trunk_model_path = trunk_path

    head_model, trunk_model = dummy_wrapper.train()
    assert head_model["value"] == 2
    assert trunk_model["value"] == -2


def test_train_fake_data(dummy_wrapper):
    head_model, trunk_model = dummy_wrapper.train(fake_data=True, n_fake_samples=2)
    assert head_model["value"] == 1
    assert trunk_model["value"] == -1


@pytest.mark.parametrize(
    "fake_data,n_fake_samples,expected_pred",
    [
        (False, 0, []),
        (True, 1, []),
    ],
)
def test_predict(fake_data, n_fake_samples, expected_pred, create_models, dummy_wrapper):
    _, head_path, trunk_path = create_models
    dummy_wrapper._workspace.input_head_model_path = head_path
    dummy_wrapper._workspace.input_trunk_model_path = trunk_path

    pred = dummy_wrapper.predict(
        fake_data=fake_data,
        n_fake_samples=n_fake_samples,
    )
    assert pred == expected_pred


def test_execute_train(workdir):
    output_models_path = workdir / "output_models"
    output_models_path.mkdir()
    output_head_model_filename = "head_model"
    output_trunk_model_filename = "trunk_model"

    output_head_model_path = output_models_path / output_head_model_filename
    assert not output_head_model_path.exists()
    output_trunk_model_path = output_models_path / output_trunk_model_filename
    assert not output_trunk_model_path.exists()

    outputs = [
        {"id": COMPOSITE_IO_LOCAL, "value": str(output_head_model_path)},
        {"id": COMPOSITE_IO_SHARED, "value": str(output_trunk_model_path)},
    ]
    inputs = [
        {"id": TASK_IO_DATASAMPLES, "value": str(workdir / "datasamples_unused")},
    ]

    common_args = [
        "--outputs",
        json.dumps(outputs),
        "--inputs",
        json.dumps(inputs),
    ]

    algo.execute(DummyCompositeAlgo(), sysargs=["train"] + common_args)
    assert output_head_model_path.exists()
    assert output_trunk_model_path.exists()


def test_execute_train_multiple_models(workdir, create_models):
    _, head_path, trunk_path = create_models

    output_models_folder_path = workdir / "output_models"
    output_models_folder_path.mkdir()

    output_head_model_filename = "output_head_model"
    output_head_model_path = output_models_folder_path / output_head_model_filename
    assert not output_head_model_path.exists()

    output_trunk_model_filename = "output_trunk_model"
    output_trunk_model_path = output_models_folder_path / output_trunk_model_filename
    assert not output_trunk_model_path.exists()

    pred_path = workdir / "pred" / "pred"
    assert not pred_path.exists()

    outputs = [
        {"id": COMPOSITE_IO_LOCAL, "value": str(output_head_model_path)},
        {"id": COMPOSITE_IO_SHARED, "value": str(output_trunk_model_path)},
    ]
    inputs = [
        {"id": TASK_IO_DATASAMPLES, "value": str(workdir / "datasamples_unused")},
        {"id": COMPOSITE_IO_LOCAL, "value": str(head_path)},
        {"id": COMPOSITE_IO_SHARED, "value": str(trunk_path)},
    ]

    command = [
        "train",
        "--outputs",
        json.dumps(outputs),
        "--inputs",
        json.dumps(inputs),
    ]

    algo.execute(DummyCompositeAlgo(), sysargs=command)
    assert output_head_model_path.exists()
    with open(output_head_model_path, "r") as f:
        head_model = json.load(f)
    assert head_model["value"] == 2

    assert output_trunk_model_path.exists()
    with open(output_trunk_model_path, "r") as f:
        trunk_model = json.load(f)
    assert trunk_model["value"] == -2

    assert not pred_path.exists()


def test_execute_predict(workdir, create_models):
    _, head_path, trunk_path = create_models

    pred_path = workdir / "pred" / "pred"
    assert not pred_path.exists()

    inputs = [
        {"id": TASK_IO_DATASAMPLES, "value": str(workdir / "datasamples_unused")},
        {"id": COMPOSITE_IO_LOCAL, "value": str(head_path)},
        {"id": COMPOSITE_IO_SHARED, "value": str(trunk_path)},
    ]

    command = [
        "predict",
        "--inputs",
        json.dumps(inputs),
    ]

    algo.execute(DummyCompositeAlgo(), sysargs=command)
    assert pred_path.exists()
    with open(pred_path, "r") as f:
        pred = json.load(f)
    assert pred == []
    pred_path.unlink()


@pytest.mark.parametrize(
    "algo_class",
    (
        NoSavedTrunkModelAggregateAlgo,
        NoSavedHeadModelAggregateAlgo,
        WrongSavedTrunkModelAggregateAlgo,
        WrongSavedHeadModelAggregateAlgo,
    ),
)
def test_model_check(algo_class, workdir, valid_composite_algo_workspace):
    a = algo_class()
    wp = algo.CompositeAlgoWrapper(a, valid_composite_algo_workspace)

    with pytest.raises(exceptions.MissingFileError):
        wp.train()
