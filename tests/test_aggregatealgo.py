import json
from os import PathLike
from typing import Any
from typing import List
from typing import TypedDict

import pytest

from substratools import algo
from substratools import exceptions
from substratools.algo import InputIdentifiers
from substratools.algo import OutputIdentifiers
from substratools.task_resources import TASK_IO_PREDICTIONS
from substratools.task_resources import TRAIN_IO_MODEL
from substratools.task_resources import TRAIN_IO_MODELS
from substratools.workspace import AggregateAlgoWorkspace
from tests import utils


@pytest.fixture(autouse=True)
def setup(valid_opener):
    pass


class DummyAggregateAlgo(algo.AggregateAlgo):
    def aggregate(
        self,
        inputs: TypedDict(
            "inputs",
            {
                InputIdentifiers.models: List[PathLike],
                InputIdentifiers.rank: int,
            },
        ),
        outputs: TypedDict("outputs", {"model": PathLike}),
    ) -> None:

        models = utils.load_models(paths=inputs.get(InputIdentifiers.models, []))

        new_model = {"value": 0}
        for m in models:
            new_model["value"] += m["value"]

        utils.save_model(model=new_model, path=outputs.get(OutputIdentifiers.model))

    def predict(
        self,
        inputs: TypedDict(
            "inputs",
            {
                InputIdentifiers.X: Any,
                "model": PathLike,
            },
        ),
        outputs: TypedDict("outputs", {"model": PathLike}),
    ):
        model = utils.load_model(path=inputs.get(InputIdentifiers.model))

        # Predict
        X = inputs.get(InputIdentifiers.X)
        pred = X * model["value"]

        # save predictions
        utils.save_predictions(predictions=pred, path=outputs.get(OutputIdentifiers.predictions))


class NoSavedModelAggregateAlgo(DummyAggregateAlgo):
    def aggregate(self, inputs, outputs):

        models = utils.load_models(paths=inputs.get(InputIdentifiers.models, []))

        new_model = {"value": 0}
        for m in models:
            new_model["value"] += m["value"]

        utils.no_save_model(model=new_model, path=outputs.get(OutputIdentifiers.model))


class WrongSavedModelAggregateAlgo(DummyAggregateAlgo):
    def aggregate(self, inputs, outputs):

        models = utils.load_models(paths=inputs.get(InputIdentifiers.models, []))

        new_model = {"value": 0}
        for m in models:
            new_model["value"] += m["value"]

        utils.wrong_save_model(model=new_model, path=outputs.get(OutputIdentifiers.model))


@pytest.fixture
def create_models(workdir):
    model_a = {"value": 1}
    model_b = {"value": 2}

    model_dir = workdir / "model"
    model_dir.mkdir()

    def _create_model(model_data):
        model_name = model_data["value"]
        filename = "{}.json".format(model_name)
        path = model_dir / filename
        path.write_text(json.dumps(model_data))
        return path

    model_datas = [model_a, model_b]
    model_filenames = [_create_model(d) for d in model_datas]

    return model_datas, model_filenames


def test_create():
    # check we can instantiate a dummy algo class
    DummyAggregateAlgo()


def test_aggregate_no_model(valid_algo_workspace):
    a = DummyAggregateAlgo()
    wp = algo.AggregateAlgoWrapper(a, valid_algo_workspace)
    wp.aggregate()
    model = utils.load_model(valid_algo_workspace.output_model_path)
    assert model["value"] == 0


def test_aggregate_multiple_models(create_models, output_model_path):
    _, model_filenames = create_models

    workspace = AggregateAlgoWorkspace(input_model_paths=model_filenames, output_model_path=output_model_path)
    a = DummyAggregateAlgo()
    wp = algo.AggregateAlgoWrapper(a, workspace)

    wp.aggregate()
    model = utils.load_model(wp._workspace.output_model_path)

    assert model["value"] == 3


@pytest.mark.parametrize(
    "fake_data,expected_pred,n_fake_samples",
    [
        (False, InputIdentifiers.X, None),
        (True, ["Xfake"], 1),
    ],
)
def test_predict(fake_data, expected_pred, n_fake_samples, create_models):
    _, model_filenames = create_models

    a = DummyAggregateAlgo()
    workspace = AggregateAlgoWorkspace(input_model_paths=[model_filenames[0]])
    wp = algo.AggregateAlgoWrapper(a, workspace)
    wp.predict(fake_data=fake_data, n_fake_samples=n_fake_samples)
    pred = utils.load_predictions(workspace.output_predictions_path)
    assert pred == expected_pred


def test_execute_aggregate(output_model_path):

    assert not output_model_path.exists()

    outputs = [{"id": TRAIN_IO_MODEL, "value": str(output_model_path)}]

    algo.execute(DummyAggregateAlgo(), sysargs=["aggregate", "--outputs", json.dumps(outputs)])
    assert output_model_path.exists()

    output_model_path.unlink()
    algo.execute(DummyAggregateAlgo(), sysargs=["aggregate", "--outputs", json.dumps(outputs), "--log-level", "debug"])
    assert output_model_path.exists()


def test_execute_aggregate_multiple_models(workdir, create_models, output_model_path):
    _, model_filenames = create_models

    assert not output_model_path.exists()

    inputs = [{"id": TRAIN_IO_MODELS, "value": str(workdir / model)} for model in model_filenames]
    outputs = [
        {"id": TRAIN_IO_MODEL, "value": str(output_model_path)},
    ]
    options = ["--inputs", json.dumps(inputs), "--outputs", json.dumps(outputs)]

    command = ["aggregate"]
    command.extend(options)

    algo.execute(DummyAggregateAlgo(), sysargs=command)
    assert output_model_path.exists()
    with open(output_model_path, "r") as f:
        model = json.load(f)
    assert model["value"] == 3


def test_execute_predict(workdir, create_models, output_model_path):
    _, model_filenames = create_models
    assert not output_model_path.exists()

    inputs = [{"id": TRAIN_IO_MODELS, "value": str(workdir / model_name)} for model_name in model_filenames]
    outputs = [{"id": TRAIN_IO_MODEL, "value": str(output_model_path)}]
    options = ["--inputs", json.dumps(inputs), "--outputs", json.dumps(outputs)]
    command = ["aggregate"]
    command.extend(options)
    algo.execute(DummyAggregateAlgo(), sysargs=command)
    assert output_model_path.exists()

    # do predict on output model
    pred_path = workdir / "pred" / "pred"
    assert not pred_path.exists()

    pred_inputs = [
        {"id": TRAIN_IO_MODELS, "value": str(output_model_path)},
    ]
    pred_outputs = [{"id": TASK_IO_PREDICTIONS, "value": str(pred_path)}]
    pred_options = ["--inputs", json.dumps(pred_inputs), "--outputs", json.dumps(pred_outputs)]

    algo.execute(DummyAggregateAlgo(), sysargs=["predict"] + pred_options)
    assert pred_path.exists()
    with open(pred_path, "r") as f:
        pred = json.load(f)
    assert pred == "XXX"
    pred_path.unlink()


@pytest.mark.parametrize("algo_class", (NoSavedModelAggregateAlgo, WrongSavedModelAggregateAlgo))
def test_model_check(algo_class, valid_algo_workspace):
    a = algo_class()
    wp = algo.AggregateAlgoWrapper(a, valid_algo_workspace)

    with pytest.raises(exceptions.MissingFileError):
        wp.aggregate([])
