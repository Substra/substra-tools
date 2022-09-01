import json
import os
from typing import Any
from typing import Optional
from typing import TypedDict

import pytest

from substratools import algo
from substratools import exceptions
from substratools.task_resources import TASK_IO_DATASAMPLES
from substratools.workspace import CompositeAlgoWorkspace
from tests import utils


@pytest.fixture(autouse=True)
def setup(valid_opener):
    pass


class DummyCompositeAlgo(algo.CompositeAlgo):
    def train(
        self,
        inputs: TypedDict(
            "inputs",
            {
                "X": Any,
                "y": Any,
                "local": Optional[os.PathLike],
                "shared": Optional[os.PathLike],
                "rank": int,
            },
        ),
        outputs: TypedDict(
            "outputs",
            {
                "local": os.PathLike,
                "shared": os.PathLike,
            },
        ),
    ):
        # init phase
        # load models
        head_model = utils.load_model(path=inputs.get("local"))
        trunk_model = utils.load_model(path=inputs.get("shared"))

        if head_model and trunk_model:
            new_head_model = dict(head_model)
            new_trunk_model = dict(trunk_model)
        else:
            new_head_model = {"value": 0}
            new_trunk_model = {"value": 0}

        # train models
        new_head_model["value"] += 1
        new_trunk_model["value"] -= 1

        # save model
        utils.save_model(model=new_head_model, path=outputs.get("local"))
        utils.save_model(model=new_trunk_model, path=outputs.get("shared"))

    def predict(
        self,
        inputs: TypedDict(
            "inputs",
            {
                "X": Any,
                "local": os.PathLike,
                "shared": os.PathLike,
            },
        ),
        outputs: TypedDict(
            "outputs",
            {
                "predictions": os.PathLike,
            },
        ),
    ):

        # init phase
        # load models
        head_model = utils.load_model(path=inputs.get("local"))
        trunk_model = utils.load_model(path=inputs.get("shared"))

        pred = list(range(head_model["value"], trunk_model["value"]))

        # save predictions
        utils.save_predictions(predictions=pred, path=outputs.get("predictions"))


class NoSavedTrunkModelAggregateAlgo(DummyCompositeAlgo):
    def train(self, inputs, outputs):
        # init phase
        # load models
        head_model = utils.load_model(path=inputs.get("local"))
        trunk_model = utils.load_model(path=inputs.get("shared"))

        if head_model and trunk_model:
            new_head_model = dict(head_model)
            new_trunk_model = dict(trunk_model)
        else:
            new_head_model = {"value": 0}
            new_trunk_model = {"value": 0}

        # train models
        new_head_model["value"] += 1
        new_trunk_model["value"] -= 1

        # save model
        utils.save_model(model=new_head_model, path=outputs.get("local"))
        utils.no_save_model(model=new_trunk_model, path=outputs.get("shared"))


class NoSavedHeadModelAggregateAlgo(DummyCompositeAlgo):
    def train(self, inputs, outputs):
        # init phase
        # load models
        head_model = utils.load_model(path=inputs.get("local"))
        trunk_model = utils.load_model(path=inputs.get("shared"))

        if head_model and trunk_model:
            new_head_model = dict(head_model)
            new_trunk_model = dict(trunk_model)
        else:
            new_head_model = {"value": 0}
            new_trunk_model = {"value": 0}

        # train models
        new_head_model["value"] += 1
        new_trunk_model["value"] -= 1

        # save model
        utils.no_save_model(model=new_head_model, path=outputs.get("local"))
        utils.save_model(model=new_trunk_model, path=outputs.get("shared"))


class WrongSavedTrunkModelAggregateAlgo(DummyCompositeAlgo):
    def train(self, inputs, outputs):
        # init phase
        # load models
        head_model = utils.load_model(path=inputs.get("local"))
        trunk_model = utils.load_model(path=inputs.get("shared"))

        if head_model and trunk_model:
            new_head_model = dict(head_model)
            new_trunk_model = dict(trunk_model)
        else:
            new_head_model = {"value": 0}
            new_trunk_model = {"value": 0}

        # train models
        new_head_model["value"] += 1
        new_trunk_model["value"] -= 1

        # save model
        utils.save_model(model=new_head_model, path=outputs.get("local"))
        utils.wrong_save_model(model=new_trunk_model, path=outputs.get("shared"))


class WrongSavedHeadModelAggregateAlgo(DummyCompositeAlgo):
    def train(self, inputs, outputs):
        # init phase
        # load models
        head_model = utils.load_model(path=inputs.get("local"))
        trunk_model = utils.load_model(path=inputs.get("shared"))

        if head_model and trunk_model:
            new_head_model = dict(head_model)
            new_trunk_model = dict(trunk_model)
        else:
            new_head_model = {"value": 0}
            new_trunk_model = {"value": 0}

        # train models
        new_head_model["value"] += 1
        new_trunk_model["value"] -= 1

        # save model
        utils.wrong_save_model(model=new_head_model, path=outputs.get("local"))
        utils.save_model(model=new_trunk_model, path=outputs.get("shared"))


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
    dummy_wrapper.train()
    head_model = utils.load_model(dummy_wrapper._workspace.output_head_model_path)
    trunk_model = utils.load_model(dummy_wrapper._workspace.output_trunk_model_path)
    assert head_model["value"] == 1
    assert trunk_model["value"] == -1


def test_train_input_head_trunk_models(create_models, dummy_wrapper):

    _, head_path, trunk_path = create_models

    dummy_wrapper._workspace.input_head_model_path = head_path
    dummy_wrapper._workspace.input_trunk_model_path = trunk_path

    dummy_wrapper.train()

    head_model = utils.load_model(dummy_wrapper._workspace.output_head_model_path)
    trunk_model = utils.load_model(dummy_wrapper._workspace.output_trunk_model_path)

    assert head_model["value"] == 2
    assert trunk_model["value"] == -2


def test_train_fake_data(dummy_wrapper):
    dummy_wrapper.train(fake_data=True, n_fake_samples=2)
    head_model = utils.load_model(dummy_wrapper._workspace.output_head_model_path)
    trunk_model = utils.load_model(dummy_wrapper._workspace.output_trunk_model_path)
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

    dummy_wrapper.predict(
        fake_data=fake_data,
        n_fake_samples=n_fake_samples,
    )

    pred = utils.load_predictions(dummy_wrapper._workspace.output_predictions_path)

    assert pred == expected_pred


def test_execute_train(workdir):
    output_models_path = workdir / "output_models"
    output_models_path.mkdir()
    output_head_model_filename = "local"
    output_trunk_model_filename = "shared"

    output_head_model_path = output_models_path / output_head_model_filename
    assert not output_head_model_path.exists()
    output_trunk_model_path = output_models_path / output_trunk_model_filename
    assert not output_trunk_model_path.exists()

    outputs = [
        {"id": "local", "value": str(output_head_model_path)},
        {"id": "shared", "value": str(output_trunk_model_path)},
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

    algo.execute(DummyCompositeAlgo(), sysargs=["--method-name", "train"] + common_args)
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
        {"id": "local", "value": str(output_head_model_path)},
        {"id": "shared", "value": str(output_trunk_model_path)},
    ]
    inputs = [
        {"id": TASK_IO_DATASAMPLES, "value": str(workdir / "datasamples_unused")},
        {"id": "local", "value": str(head_path)},
        {"id": "shared", "value": str(trunk_path)},
    ]

    command = [
        "--method-name",
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
        {"id": "local", "value": str(head_path)},
        {"id": "shared", "value": str(trunk_path)},
    ]

    command = [
        "--method-name",
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
