import json
from os import PathLike
from typing import Any
from typing import List
from typing import TypedDict
from uuid import uuid4

import pytest

from substratools import algo
from substratools import exceptions
from substratools import opener
from substratools.task_resources import TaskResources
from substratools.workspace import AggregateAlgoWorkspace
from tests.utils import InputIdentifiers
from tests.utils import OutputIdentifiers
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
        outputs: TypedDict("outputs", {OutputIdentifiers.model: PathLike}),
    ) -> None:
        if inputs is not None:
            models = utils.load_models(paths=inputs.get(InputIdentifiers.models, []))
        else:
            models = []

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
                InputIdentifiers.model: PathLike,
            },
        ),
        outputs: TypedDict("outputs", {OutputIdentifiers.model: PathLike}),
    ):
        model = utils.load_model(path=inputs.get(OutputIdentifiers.model))

        # Predict
        X = inputs.get(InputIdentifiers.X)
        pred = X * model["value"]

        # save predictions
        utils.save_predictions(predictions=pred, path=outputs.get(OutputIdentifiers.predictions))


class NoSavedModelAggregateAlgo(DummyAggregateAlgo):
    def aggregate(self, inputs, outputs):

        if inputs is not None:
            models = utils.load_models(paths=inputs.get(InputIdentifiers.models, []))
        else:
            models = []

        new_model = {"value": 0}
        for m in models:
            new_model["value"] += m["value"]

        utils.no_save_model(model=new_model, path=outputs.get(OutputIdentifiers.model))


class WrongSavedModelAggregateAlgo(DummyAggregateAlgo):
    def aggregate(self, inputs, outputs):

        if inputs is not None:
            models = utils.load_models(paths=inputs.get(InputIdentifiers.models, []))
        else:
            models = []

        new_model = {"value": 0}
        for m in models:
            new_model["value"] += m["value"]

        utils.wrong_save_model(model=new_model, path=outputs.get(OutputIdentifiers.model))


@pytest.fixture
def create_models(workdir):
    model_a = {"value": 1}
    model_b = {"value": 2}

    model_dir = workdir / OutputIdentifiers.model
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
    DummyAggregateAlgo()


def test_aggregate_no_model(valid_algo_workspace):
    a = DummyAggregateAlgo()
    wp = algo.GenericAlgoWrapper(a, valid_algo_workspace, opener_wrapper=None)
    wp.task_launcher(method_name="aggregate")
    model = utils.load_model(wp._workspace.task_outputs[OutputIdentifiers.model])
    assert model["value"] == 0


def test_aggregate_multiple_models(create_models, output_model_path):
    _, model_filenames = create_models

    workspace_inputs = TaskResources(
        json.dumps([{"id": InputIdentifiers.models, "value": f, "multiple": True} for f in model_filenames])
    )
    workspace_outputs = TaskResources(
        json.dumps([{"id": OutputIdentifiers.model, "value": str(output_model_path), "multiple": False}])
    )

    workspace = AggregateAlgoWorkspace(inputs=workspace_inputs, outputs=workspace_outputs)
    a = DummyAggregateAlgo()
    wp = algo.GenericAlgoWrapper(a, workspace, opener_wrapper=None)

    wp.task_launcher(method_name="aggregate")
    model = utils.load_model(wp._workspace.task_outputs[OutputIdentifiers.model])

    assert model["value"] == 3


@pytest.mark.parametrize(
    "fake_data,expected_pred,n_fake_samples",
    [
        (False, "X", None),
        (True, ["Xfake"], 1),
    ],
)
def test_predict(fake_data, expected_pred, n_fake_samples, create_models):
    _, model_filenames = create_models

    workspace_inputs = TaskResources(
        json.dumps([{"id": InputIdentifiers.model, "value": model_filenames[0], "multiple": False}])
    )
    workspace_outputs = TaskResources(
        json.dumps([{"id": OutputIdentifiers.predictions, "value": model_filenames[0], "multiple": False}])
    )

    workspace = AggregateAlgoWorkspace(inputs=workspace_inputs, outputs=workspace_outputs)
    a = DummyAggregateAlgo()

    wp = algo.GenericAlgoWrapper(a, workspace, opener_wrapper=opener.load_from_module())

    wp.task_launcher(method_name="predict", fake_data=fake_data, n_fake_samples=n_fake_samples)

    pred = utils.load_predictions(wp._workspace.task_outputs[OutputIdentifiers.predictions])
    assert pred == expected_pred


def test_execute_aggregate(output_model_path):

    assert not output_model_path.exists()

    outputs = [{"id": OutputIdentifiers.model, "value": str(output_model_path), "multiple": False}]

    algo.execute(DummyAggregateAlgo(), sysargs=["--method-name", "aggregate", "--outputs", json.dumps(outputs)])
    assert output_model_path.exists()

    output_model_path.unlink()
    algo.execute(
        DummyAggregateAlgo(),
        sysargs=["--method-name", "aggregate", "--outputs", json.dumps(outputs), "--log-level", "debug"],
    )
    assert output_model_path.exists()


def test_execute_aggregate_multiple_models(workdir, create_models, output_model_path):
    _, model_filenames = create_models

    assert not output_model_path.exists()

    inputs = [
        {"id": InputIdentifiers.models, "value": str(workdir / model), "multiple": True} for model in model_filenames
    ]
    outputs = [
        {"id": OutputIdentifiers.model, "value": str(output_model_path), "multiple": False},
    ]
    options = ["--inputs", json.dumps(inputs), "--outputs", json.dumps(outputs)]

    command = ["--method-name", "aggregate"]
    command.extend(options)

    algo.execute(DummyAggregateAlgo(), sysargs=command)
    assert output_model_path.exists()
    with open(output_model_path, "r") as f:
        model = json.load(f)
    assert model["value"] == 3


def test_execute_predict(workdir, create_models, output_model_path, valid_opener_script):
    _, model_filenames = create_models
    assert not output_model_path.exists()

    inputs = [
        {"id": InputIdentifiers.models, "value": str(workdir / model_name), "multiple": True}
        for model_name in model_filenames
    ]
    outputs = [{"id": OutputIdentifiers.model, "value": str(output_model_path), "multiple": False}]
    options = ["--inputs", json.dumps(inputs), "--outputs", json.dumps(outputs)]
    command = ["--method-name", "aggregate"]
    command.extend(options)
    algo.execute(DummyAggregateAlgo(), sysargs=command)
    assert output_model_path.exists()

    # do predict on output model
    pred_path = workdir / str(uuid4())
    assert not pred_path.exists()

    pred_inputs = [
        {"id": InputIdentifiers.model, "value": str(output_model_path), "multiple": False},
        {"id": InputIdentifiers.opener, "value": valid_opener_script, "multiple": False},
    ]
    pred_outputs = [{"id": OutputIdentifiers.predictions, "value": str(pred_path), "multiple": False}]
    pred_options = ["--inputs", json.dumps(pred_inputs), "--outputs", json.dumps(pred_outputs)]

    algo.execute(DummyAggregateAlgo(), sysargs=["--method-name", "predict"] + pred_options)
    assert pred_path.exists()
    with open(pred_path, "r") as f:
        pred = json.load(f)
    assert pred == "XXX"
    pred_path.unlink()


@pytest.mark.parametrize("algo_class", (NoSavedModelAggregateAlgo, WrongSavedModelAggregateAlgo))
def test_model_check(algo_class, valid_algo_workspace):
    a = algo_class()
    wp = algo.GenericAlgoWrapper(a, valid_algo_workspace, opener_wrapper=None)

    with pytest.raises(exceptions.MissingFileError):
        wp.task_launcher(method_name="aggregate")
