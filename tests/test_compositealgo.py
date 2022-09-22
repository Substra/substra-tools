import json
import os

from typing import Any
from typing import Optional
from typing import TypedDict
import pytest

from substratools import algo
from substratools import exceptions
from substratools.task_resources import StaticInputIdentifiers
from substratools.task_resources import TaskResources
from substratools.workspace import AlgoWorkspace
from substratools import opener
from tests import utils
from tests.utils import InputIdentifiers
from tests.utils import OutputIdentifiers


@pytest.fixture(autouse=True)
def setup(valid_opener):
    pass


class FakeDataAlgo(algo.CompositeAlgo):
    def train(self, inputs: dict, outputs: dict, task_properties: dict):
        utils.save_model(model=inputs[InputIdentifiers.datasamples][0], path=outputs["local"])
        utils.save_model(model=inputs[InputIdentifiers.datasamples][1], path=outputs["shared"])

    def predict(self, inputs: dict, outputs: dict, task_properties: dict) -> None:
        utils.save_model(model=inputs[InputIdentifiers.datasamples][0], path=outputs["predictions"])


class DummyCompositeAlgo(algo.CompositeAlgo):
    def train(
        self,
        inputs: TypedDict(
            "inputs",
            {
                InputIdentifiers.datasamples: Any,
                InputIdentifiers.local: Optional[os.PathLike],
                InputIdentifiers.shared: Optional[os.PathLike],
            },
        ),
        outputs: TypedDict(
            "outputs",
            {
                OutputIdentifiers.local: os.PathLike,
                OutputIdentifiers.shared: os.PathLike,
            },
        ),
        task_properties: TypedDict("task_properties", {InputIdentifiers.rank: int}),
    ):
        # init phase
        # load models
        head_model = utils.load_model(path=inputs.get(InputIdentifiers.local))
        trunk_model = utils.load_model(path=inputs.get(InputIdentifiers.shared))

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
        utils.save_model(model=new_head_model, path=outputs.get(OutputIdentifiers.local))
        utils.save_model(model=new_trunk_model, path=outputs.get(OutputIdentifiers.shared))

    def predict(
        self,
        inputs: TypedDict(
            "inputs",
            {
                InputIdentifiers.datasamples: Any,
                InputIdentifiers.local: os.PathLike,
                InputIdentifiers.shared: os.PathLike,
            },
        ),
        outputs: TypedDict(
            "outputs",
            {
                OutputIdentifiers.predictions: os.PathLike,
            },
        ),
    ):

        # init phase
        # load models
        head_model = utils.load_model(path=inputs.get(InputIdentifiers.local))
        trunk_model = utils.load_model(path=inputs.get(InputIdentifiers.shared))

        pred = list(range(head_model["value"], trunk_model["value"]))

        # save predictions
        utils.save_predictions(predictions=pred, path=outputs.get(OutputIdentifiers.predictions))


class NoSavedTrunkModelAggregateAlgo(DummyCompositeAlgo):
    def train(self, inputs, outputs, task_properties):
        # init phase
        # load models
        head_model = utils.load_model(path=inputs.get(InputIdentifiers.local))
        trunk_model = utils.load_model(path=inputs.get(InputIdentifiers.shared))

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
        utils.save_model(model=new_head_model, path=outputs.get(OutputIdentifiers.local))
        utils.no_save_model(model=new_trunk_model, path=outputs.get(OutputIdentifiers.shared))


class NoSavedHeadModelAggregateAlgo(DummyCompositeAlgo):
    def train(self, inputs, outputs, task_properties):
        # init phase
        # load models
        head_model = utils.load_model(path=inputs.get(InputIdentifiers.local))
        trunk_model = utils.load_model(path=inputs.get(InputIdentifiers.shared))

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
        utils.no_save_model(model=new_head_model, path=outputs.get(OutputIdentifiers.local))
        utils.save_model(model=new_trunk_model, path=outputs.get(OutputIdentifiers.shared))


class WrongSavedTrunkModelAggregateAlgo(DummyCompositeAlgo):
    def train(self, inputs, outputs, task_properties):
        # init phase
        # load models
        head_model = utils.load_model(path=inputs.get(InputIdentifiers.local))
        trunk_model = utils.load_model(path=inputs.get(InputIdentifiers.shared))

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
        utils.save_model(model=new_head_model, path=outputs.get(OutputIdentifiers.local))
        utils.wrong_save_model(model=new_trunk_model, path=outputs.get(OutputIdentifiers.shared))


class WrongSavedHeadModelAggregateAlgo(DummyCompositeAlgo):
    def train(self, inputs, outputs, task_properties):
        # init phase
        # load models
        head_model = utils.load_model(path=inputs.get(InputIdentifiers.local))
        trunk_model = utils.load_model(path=inputs.get(InputIdentifiers.shared))

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
        utils.wrong_save_model(model=new_head_model, path=outputs.get(OutputIdentifiers.local))
        utils.save_model(model=new_trunk_model, path=outputs.get(OutputIdentifiers.shared))


@pytest.fixture
def train_outputs(output_model_path, output_model_path_2):
    outputs = TaskResources(
        json.dumps(
            [
                {"id": "local", "value": str(output_model_path), "multiple": False},
                {"id": "shared", "value": str(output_model_path_2), "multiple": False},
            ]
        )
    )
    return outputs


@pytest.fixture
def composite_inputs(create_models):
    _, local_path, shared_path = create_models
    inputs = TaskResources(
        json.dumps(
            [
                {"id": InputIdentifiers.local, "value": str(local_path), "multiple": False},
                {"id": InputIdentifiers.shared, "value": str(shared_path), "multiple": False},
            ]
        )
    )

    return inputs


@pytest.fixture
def predict_outputs(output_model_path):
    outputs = TaskResources(
        json.dumps([{"id": OutputIdentifiers.predictions, "value": str(output_model_path), "multiple": False}])
    )
    return outputs


@pytest.fixture
def create_models(workdir):
    head_model = {"value": 1}
    trunk_model = {"value": -1}

    def _create_model(model_data, name):
        filename = "{}.json".format(name)
        path = workdir / filename
        path.write_text(json.dumps(model_data))
        return path

    head_path = _create_model(head_model, "head")
    trunk_path = _create_model(trunk_model, "trunk")

    return (
        [head_model, trunk_model],
        head_path,
        trunk_path,
    )


def test_create():
    # check we can instantiate a dummy algo class
    DummyCompositeAlgo()


def test_train_no_model(train_outputs):

    a = DummyCompositeAlgo()
    dummy_train_workspace = AlgoWorkspace(outputs=train_outputs)
    dummy_train_wrapper = algo.GenericAlgoWrapper(a, dummy_train_workspace, None)
    dummy_train_wrapper.execute(method_name="train")
    local_model = utils.load_model(dummy_train_wrapper._workspace.task_outputs["local"])
    shared_model = utils.load_model(dummy_train_wrapper._workspace.task_outputs["shared"])

    assert local_model["value"] == 1
    assert shared_model["value"] == -1


def test_train_input_head_trunk_models(composite_inputs, train_outputs):

    a = DummyCompositeAlgo()
    dummy_train_workspace = AlgoWorkspace(inputs=composite_inputs, outputs=train_outputs)
    dummy_train_wrapper = algo.GenericAlgoWrapper(a, dummy_train_workspace, None)
    dummy_train_wrapper.execute(method_name="train")
    local_model = utils.load_model(dummy_train_wrapper._workspace.task_outputs["local"])
    shared_model = utils.load_model(dummy_train_wrapper._workspace.task_outputs["shared"])

    assert local_model["value"] == 2
    assert shared_model["value"] == -2


@pytest.mark.parametrize("n_fake_samples", (0, 1, 2))
def test_train_fake_data(train_outputs, n_fake_samples):
    a = FakeDataAlgo()
    _opener = opener.load_from_module()
    dummy_train_workspace = AlgoWorkspace(outputs=train_outputs)
    dummy_train_wrapper = algo.GenericAlgoWrapper(a, dummy_train_workspace, _opener)
    dummy_train_wrapper.execute(method_name="train", fake_data=bool(n_fake_samples), n_fake_samples=n_fake_samples)

    local_model = utils.load_model(dummy_train_wrapper._workspace.task_outputs[OutputIdentifiers.local])
    shared_model = utils.load_model(dummy_train_wrapper._workspace.task_outputs[OutputIdentifiers.shared])

    assert local_model == _opener.get_data(fake_data=bool(n_fake_samples), n_fake_samples=n_fake_samples)[0]
    assert shared_model == _opener.get_data(fake_data=bool(n_fake_samples), n_fake_samples=n_fake_samples)[1]


@pytest.mark.parametrize("n_fake_samples", (0, 1, 2))
def test_predict_fake_data(composite_inputs, predict_outputs, n_fake_samples):
    a = FakeDataAlgo()
    _opener = opener.load_from_module()
    dummy_train_workspace = AlgoWorkspace(inputs=composite_inputs, outputs=predict_outputs)
    dummy_train_wrapper = algo.GenericAlgoWrapper(a, dummy_train_workspace, _opener)
    dummy_train_wrapper.execute(method_name="predict", fake_data=bool(n_fake_samples), n_fake_samples=n_fake_samples)

    predictions = utils.load_model(dummy_train_wrapper._workspace.task_outputs[OutputIdentifiers.predictions])

    assert predictions == _opener.get_data(fake_data=bool(n_fake_samples), n_fake_samples=n_fake_samples)[0]


@pytest.mark.parametrize(
    "algo_class",
    (
        NoSavedTrunkModelAggregateAlgo,
        NoSavedHeadModelAggregateAlgo,
        WrongSavedTrunkModelAggregateAlgo,
        WrongSavedHeadModelAggregateAlgo,
    ),
)
def test_model_check(algo_class, train_outputs):
    a = algo_class()
    dummy_train_workspace = AlgoWorkspace(outputs=train_outputs)
    wp = algo.GenericAlgoWrapper(interface=a, workspace=dummy_train_workspace, opener_wrapper=None)

    with pytest.raises(exceptions.MissingFileError):
        wp.execute("train")
