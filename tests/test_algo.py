import json
import shutil
from os import PathLike
from typing import Any
from typing import List
from typing import Optional
from typing import TypedDict

import pytest

from substratools import algo
from substratools import exceptions
from substratools.algo import InputIdentifiers
from substratools.algo import OutputIdentifiers
from substratools.task_resources import TASK_IO_DATASAMPLES
from substratools.task_resources import TASK_IO_PREDICTIONS
from substratools.task_resources import TRAIN_IO_MODEL
from substratools.task_resources import TRAIN_IO_MODELS
from substratools.workspace import AlgoWorkspace
from tests import utils


@pytest.fixture(autouse=True)
def setup(valid_opener):
    pass


class DummyAlgo(algo.Algo):
    def train(
        self,
        inputs: TypedDict(
            "inputs",
            {
                InputIdentifiers.X: List["str"],  # cf valid_opener_code # TODO: rename "data" , del Y
                InputIdentifiers.y: List[int],  # datasamples contains loaded datasamples, if any, or None
                InputIdentifiers.models: Optional[
                    PathLike
                ],  # inputs contains a dict where keys are identifiers and values are paths on the disk
                InputIdentifiers.rank: int,
            },
        ),
        outputs: TypedDict(
            "outputs", {OutputIdentifiers.model: PathLike}
        ),  # outputs contains a dict where keys are identifiers and values are paths on disk
    ) -> None:
        # TODO: checks on data
        # load models
        models = utils.load_models(paths=inputs.get(InputIdentifiers.models, []))

        # init model
        new_model = {"value": 0}

        # train (just add the models values)
        for m in models:
            assert isinstance(m, dict)
            assert "value" in m
            new_model["value"] += m["value"]

        # save model
        utils.save_model(model=new_model, path=outputs.get(OutputIdentifiers.model))

    def predict(
        self,
        inputs: TypedDict("inputs", {InputIdentifiers.X: Any, InputIdentifiers.model: List[PathLike]}),
        outputs: TypedDict("outputs", {OutputIdentifiers.predictions: PathLike}),
    ) -> None:
        # TODO: checks on data

        # load_model
        model = utils.load_model(path=inputs.get(InputIdentifiers.model))

        # predict
        X = inputs.get(InputIdentifiers.X)
        pred = X * model["value"]

        # save predictions
        utils.save_predictions(predictions=pred, path=outputs.get(OutputIdentifiers.predictions))


class NoSavedModelAlgo(DummyAlgo):
    def train(self, inputs, outputs):
        # TODO: checks on data
        # load models
        models = utils.load_models(paths=inputs.get(InputIdentifiers.models, []))

        # init model
        new_model = {"value": 0}

        # train (just add the models values)
        for m in models:
            assert isinstance(m, dict)
            assert "value" in m
            new_model["value"] += m["value"]

        # save model
        utils.no_save_model(model=new_model, path=outputs.get(OutputIdentifiers.model))


class WrongSavedModelAlgo(DummyAlgo):
    def train(self, inputs, outputs):
        # TODO: checks on data
        # load models
        models = utils.load_models(paths=inputs.get(InputIdentifiers.models, []))

        # init model
        new_model = {"value": 0}

        # train (just add the models values)
        for m in models:
            assert isinstance(m, dict)
            assert "value" in m
            new_model["value"] += m["value"]

        # save model
        utils.wrong_save_model(model=new_model, path=outputs.get("model"))


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
    DummyAlgo()


def test_train_no_model(valid_algo_workspace):
    a = DummyAlgo()
    wp = algo.AlgoWrapper(a, valid_algo_workspace)
    wp.train()
    model = utils.load_model(valid_algo_workspace.output_model_path)
    assert model["value"] == 0


def test_train_multiple_models(output_model_path, create_models):
    _, model_filenames = create_models

    workspace = AlgoWorkspace(input_model_paths=model_filenames, output_model_path=output_model_path)
    a = DummyAlgo()
    wp = algo.AlgoWrapper(a, workspace=workspace)

    wp.train()
    model = utils.load_model(workspace.output_model_path)

    assert model["value"] == 3


def test_train_fake_data(output_model_path):
    a = DummyAlgo()
    workspace = AlgoWorkspace(input_model_paths=[], output_model_path=output_model_path)
    wp = algo.AlgoWrapper(a, workspace=workspace)
    wp.train(fake_data=True, n_fake_samples=2)
    model = utils.load_model(workspace.output_model_path)
    assert model["value"] == 0


@pytest.mark.parametrize(
    "fake_data,expected_pred,n_fake_samples",
    [
        (False, InputIdentifiers.X, None),
        (True, ["Xfake"], 1),
    ],
)
def test_predict(fake_data, expected_pred, n_fake_samples, create_models):
    _, model_filenames = create_models

    a = DummyAlgo()

    workspace = AlgoWorkspace(input_model_paths=[model_filenames[0]])
    wp = algo.AlgoWrapper(a, workspace=workspace)
    wp.predict(fake_data=fake_data, n_fake_samples=n_fake_samples)

    pred = utils.load_predictions(workspace.output_predictions_path)
    assert pred == expected_pred


def test_execute_train(workdir, output_model_path):
    inputs = [
        {"id": TASK_IO_DATASAMPLES, "value": str(workdir / "datasamples_unused")},
    ]
    outputs = [
        {"id": TRAIN_IO_MODEL, "value": str(output_model_path)},
    ]
    options = ["--inputs", json.dumps(inputs), "--outputs", json.dumps(outputs)]

    assert not output_model_path.exists()

    algo.execute(DummyAlgo(), sysargs=["--method-name", "train"] + options)
    assert output_model_path.exists()

    algo.execute(
        DummyAlgo(),
        sysargs=["--method-name", "train", "--fake-data", "--n-fake-samples", "1", "--outputs", json.dumps(outputs)],
    )
    assert output_model_path.exists()

    algo.execute(DummyAlgo(), sysargs=["--method-name", "train", "--log-level", "debug"] + options)
    assert output_model_path.exists()


def test_execute_train_multiple_models(workdir, output_model_path, create_models):
    _, model_filenames = create_models

    assert not output_model_path.exists()
    pred_path = workdir / "pred" / "pred"
    assert not pred_path.exists()

    inputs = [
        {"id": TASK_IO_DATASAMPLES, "value": str(workdir / "datasamples_unused")},
    ] + [{"id": TRAIN_IO_MODELS, "value": str(workdir / model)} for model in model_filenames]
    outputs = [
        {"id": TRAIN_IO_MODEL, "value": str(output_model_path)},
    ]
    options = ["--inputs", json.dumps(inputs), "--outputs", json.dumps(outputs)]

    command = ["--method-name", "train"]
    command.extend(options)

    algo.execute(DummyAlgo(), sysargs=command)
    assert output_model_path.exists()
    with open(output_model_path, "r") as f:
        model = json.load(f)
    assert model["value"] == 3

    assert not pred_path.exists()


def test_execute_predict(workdir, output_model_path, create_models):
    _, model_filenames = create_models
    pred_path = workdir / "pred" / "pred"
    train_inputs = [
        {"id": TASK_IO_DATASAMPLES, "value": str(workdir / "datasamples_unused")},
    ] + [{"id": TRAIN_IO_MODELS, "value": str(workdir / model)} for model in model_filenames]
    train_outputs = [{"id": TRAIN_IO_MODEL, "value": str(output_model_path)}]
    train_options = ["--inputs", json.dumps(train_inputs), "--outputs", json.dumps(train_outputs)]

    # first train models
    assert not pred_path.exists()
    command = ["--method-name", "train"]
    command.extend(train_options)
    algo.execute(DummyAlgo(), sysargs=command)
    assert output_model_path.exists()

    # do predict on output model
    pred_inputs = [
        {"id": TASK_IO_DATASAMPLES, "value": str(workdir / "datasamples_unused")},
        {"id": TRAIN_IO_MODELS, "value": str(output_model_path)},
    ]
    pred_outputs = [{"id": TASK_IO_PREDICTIONS, "value": str(pred_path)}]
    pred_options = ["--inputs", json.dumps(pred_inputs), "--outputs", json.dumps(pred_outputs)]

    assert not pred_path.exists()
    algo.execute(DummyAlgo(), sysargs=["--method-name", "predict"] + pred_options)
    assert pred_path.exists()
    with open(pred_path, "r") as f:
        pred = json.load(f)
    assert pred == "XXX"
    pred_path.unlink()

    # do predict with different model paths
    input_models_dir = workdir / "other_models"
    input_models_dir.mkdir()
    input_model_path = input_models_dir / "supermodel"
    shutil.move(output_model_path, input_model_path)

    pred_inputs = [
        {"id": TASK_IO_DATASAMPLES, "value": str(workdir / "datasamples_unused")},
        {"id": TRAIN_IO_MODELS, "value": str(input_model_path)},
    ]
    pred_outputs = [{"id": TASK_IO_PREDICTIONS, "value": str(pred_path)}]
    pred_options = ["--inputs", json.dumps(pred_inputs), "--outputs", json.dumps(pred_outputs)]

    assert not pred_path.exists()
    algo.execute(
        DummyAlgo(),
        sysargs=["--method-name", "predict"] + pred_options,
    )
    assert pred_path.exists()
    with open(pred_path, "r") as f:
        pred = json.load(f)
    assert pred == "XXX"


@pytest.mark.parametrize("algo_class", (NoSavedModelAlgo, WrongSavedModelAlgo))
def test_model_check(valid_algo_workspace, algo_class):
    a = algo_class()
    wp = algo.AlgoWrapper(a, workspace=valid_algo_workspace)

    with pytest.raises(exceptions.MissingFileError):
        wp.train()
