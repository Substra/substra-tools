import json
import shutil
from os import PathLike
from pathlib import Path
from typing import Any, Tuple
from typing import List
from typing import Optional
from typing import TypedDict

import pytest

from substratools import algo
from substratools import exceptions
from substratools import opener
from substratools.task_resources import StaticInputIdentifiers
from substratools.task_resources import TaskResources
from substratools.workspace import AlgoWorkspace
from tests import utils
from tests.utils import InputIdentifiers
from tests.utils import OutputIdentifiers


@pytest.fixture(autouse=True)
def setup(valid_opener):
    pass


class DummyAlgo(algo.Algo):
    def train(
        self,
        inputs: TypedDict(
            "inputs",
            {
                InputIdentifiers.datasamples: Tuple[List["str"], List[int]],  # cf valid_opener_code
                InputIdentifiers.models: Optional[
                    PathLike
                ],  # inputs contains a dict where keys are identifiers and values are paths on the disk
            },
        ),
        outputs: TypedDict(
            "outputs", {OutputIdentifiers.model: PathLike}
        ),  # outputs contains a dict where keys are identifiers and values are paths on disk
        task_properties: TypedDict("task_properties", {InputIdentifiers.rank: int}),
    ) -> None:
        # TODO: checks on data
        # load models
        if inputs:
            models = utils.load_models(paths=inputs.get(InputIdentifiers.models, []))
        else:
            models = []
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
        inputs: TypedDict("inputs", {InputIdentifiers.datasamples: Any, InputIdentifiers.model: List[PathLike]}),
        outputs: TypedDict("outputs", {OutputIdentifiers.predictions: PathLike}),
        task_properties: TypedDict("task_properties", {InputIdentifiers.rank: int}),
    ) -> None:
        # TODO: checks on data

        # load_model
        model = utils.load_model(path=inputs.get(InputIdentifiers.model))

        # predict
        X = inputs.get(InputIdentifiers.datasamples)[0]
        pred = X * model["value"]

        # save predictions
        utils.save_predictions(predictions=pred, path=outputs.get(OutputIdentifiers.predictions))


class NoSavedModelAlgo(DummyAlgo):
    def train(self, inputs, outputs, task_properties):
        # TODO: checks on data
        # load models
        if inputs:
            models = utils.load_models(paths=inputs.get(InputIdentifiers.models, []))
        else:
            models = []
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
    def train(self, inputs, outputs, task_properties):
        # TODO: checks on data
        # load models
        if inputs:
            models = utils.load_models(paths=inputs.get(InputIdentifiers.models, []))
        else:
            models = []
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
        return str(path)

    model_datas = [model_a, model_b]
    model_filenames = [_create_model(d) for d in model_datas]

    return model_datas, model_filenames


def test_create():
    # check we can instantiate a dummy algo class
    DummyAlgo()


def test_train_no_model(valid_algo_workspace):
    a = DummyAlgo()
    wp = algo.GenericAlgoWrapper(a, valid_algo_workspace, opener_wrapper=None)
    wp.execute(method_name="train")
    model = utils.load_model(wp._workspace.task_outputs[OutputIdentifiers.model])
    assert model["value"] == 0


def test_train_multiple_models(output_model_path, create_models):
    _, model_filenames = create_models

    workspace_inputs = TaskResources(
        json.dumps([{"id": InputIdentifiers.models, "value": str(f), "multiple": True} for f in model_filenames])
    )
    workspace_outputs = TaskResources(
        json.dumps([{"id": OutputIdentifiers.model, "value": str(output_model_path), "multiple": False}])
    )

    workspace = AlgoWorkspace(inputs=workspace_inputs, outputs=workspace_outputs)
    a = DummyAlgo()
    wp = algo.GenericAlgoWrapper(a, workspace=workspace, opener_wrapper=None)

    wp.execute(method_name="train")
    model = utils.load_model(wp._workspace.task_outputs[OutputIdentifiers.model])

    assert model["value"] == 3


def test_train_fake_data(output_model_path):
    a = DummyAlgo()

    workspace_outputs = TaskResources(
        json.dumps([{"id": OutputIdentifiers.model, "value": str(output_model_path), "multiple": False}])
    )

    workspace = AlgoWorkspace(outputs=workspace_outputs)
    wp = algo.GenericAlgoWrapper(a, workspace=workspace, opener_wrapper=None)
    wp.execute(method_name="train", fake_data=True, n_fake_samples=2)
    model = utils.load_model(wp._workspace.task_outputs[OutputIdentifiers.model])
    assert model["value"] == 0


@pytest.mark.parametrize(
    "fake_data,expected_pred,n_fake_samples",
    [
        (False, "X", None),
        (True, ["Xfake"], 1),
    ],
)
def test_predict(fake_data, expected_pred, n_fake_samples, create_models, output_model_path):
    _, model_filenames = create_models

    a = DummyAlgo()

    workspace_inputs = TaskResources(
        json.dumps([{"id": InputIdentifiers.model, "value": model_filenames[0], "multiple": False}])
    )
    workspace_outputs = TaskResources(
        json.dumps([{"id": OutputIdentifiers.predictions, "value": str(output_model_path), "multiple": False}])
    )

    workspace = AlgoWorkspace(inputs=workspace_inputs, outputs=workspace_outputs)
    wp = algo.GenericAlgoWrapper(a, workspace=workspace, opener_wrapper=opener.load_from_module())
    wp.execute(method_name="predict", fake_data=fake_data, n_fake_samples=n_fake_samples)

    pred = utils.load_predictions(wp._workspace.task_outputs["predictions"])
    assert pred == expected_pred


def test_execute_train(workdir, output_model_path):
    inputs = [
        {
            "id": StaticInputIdentifiers.datasamples.value,
            "value": str(workdir / "datasamples_unused"),
            "multiple": True,
        },
    ]
    outputs = [
        {"id": OutputIdentifiers.model, "value": str(output_model_path), "multiple": False},
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

    output_model_path = Path(output_model_path)

    assert not output_model_path.exists()
    pred_path = workdir / "pred"
    assert not pred_path.exists()

    inputs = [
        {"id": InputIdentifiers.models, "value": str(workdir / model), "multiple": True} for model in model_filenames
    ]

    outputs = [
        {"id": OutputIdentifiers.model, "value": str(output_model_path), "multiple": False},
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


def test_execute_predict(workdir, output_model_path, create_models, valid_opener_script):
    _, model_filenames = create_models
    pred_path = workdir / "pred"
    train_inputs = [
        {"id": InputIdentifiers.models, "value": str(workdir / model), "multiple": True} for model in model_filenames
    ]

    train_outputs = [{"id": OutputIdentifiers.model, "value": str(output_model_path), "multiple": False}]
    train_options = ["--inputs", json.dumps(train_inputs), "--outputs", json.dumps(train_outputs)]

    output_model_path = Path(output_model_path)
    # first train models
    assert not pred_path.exists()
    command = ["--method-name", "train"]
    command.extend(train_options)
    algo.execute(DummyAlgo(), sysargs=command)
    assert output_model_path.exists()

    # do predict on output model
    pred_inputs = [
        {"id": InputIdentifiers.opener, "value": valid_opener_script, "multiple": False},
        {"id": InputIdentifiers.model, "value": str(output_model_path), "multiple": False},
    ]
    pred_outputs = [{"id": OutputIdentifiers.predictions, "value": str(pred_path), "multiple": False}]
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
        {"id": InputIdentifiers.model, "value": str(input_model_path), "multiple": False},
        {"id": InputIdentifiers.opener, "value": valid_opener_script, "multiple": False},
    ]
    pred_outputs = [{"id": OutputIdentifiers.predictions, "value": str(pred_path), "multiple": False}]
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
    wp = algo.GenericAlgoWrapper(a, workspace=valid_algo_workspace, opener_wrapper=None)

    with pytest.raises(exceptions.MissingFileError):
        wp.execute(method_name="train")
