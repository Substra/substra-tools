import json
import shutil
import types

import pytest

from substratools import algo
from substratools import exceptions


@pytest.fixture(autouse=True)
def setup(valid_opener):
    pass


class DummyAlgo(algo.Algo):
    def train(self, X, y, models, rank):
        new_model = {"value": 0}
        for m in models:
            assert isinstance(m, dict)
            assert "value" in m
            new_model["value"] += m["value"]
        return new_model

    def predict(self, X, model):
        pred = model["value"]
        return X * pred

    def load_model(self, path):
        with open(path, "r") as f:
            return json.load(f)

    def save_model(self, model, path):
        with open(path, "w") as f:
            json.dump(model, f)


class NoSavedModelAlgo(DummyAlgo):
    def save_model(self, model, path):
        # do not save model at all
        pass


class WrongSavedModelAlgo(DummyAlgo):
    def save_model(self, model, path):
        # simulate numpy.save behavior
        with open(path + ".npy", "w") as f:
            json.dump(model, f)


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
        return filename

    model_datas = [model_a, model_b]
    model_filenames = [_create_model(d) for d in model_datas]

    return model_datas, model_filenames


def test_create():
    # check we can instantiate a dummy algo class
    DummyAlgo()


def test_train_no_model():
    a = DummyAlgo()
    wp = algo.AlgoWrapper(a)
    model = wp.train([])
    assert model["value"] == 0


def test_train_multiple_models(workdir, create_models):
    _, model_filenames = create_models

    a = DummyAlgo()
    wp = algo.AlgoWrapper(a)

    model = wp.train(model_filenames)
    assert model["value"] == 3


def test_train_fake_data():
    a = DummyAlgo()
    wp = algo.AlgoWrapper(a)
    model = wp.train([], fake_data=True, n_fake_samples=2)
    assert model["value"] == 0


@pytest.mark.parametrize(
    "fake_data,expected_pred,n_fake_samples",
    [
        (False, "X", None),
        (True, ["Xfake"], 1),
    ],
)
def test_predict(fake_data, expected_pred, n_fake_samples, workdir, create_models):
    _, model_filenames = create_models

    a = DummyAlgo()
    wp = algo.AlgoWrapper(a)
    pred = wp.predict(model_filenames[0], fake_data=fake_data, n_fake_samples=n_fake_samples)
    assert pred == expected_pred


def test_execute_train(workdir):

    output_model_path = workdir / "model" / "model"
    assert not output_model_path.exists()

    algo.execute(DummyAlgo(), sysargs=["train"])
    assert output_model_path.exists()

    algo.execute(DummyAlgo(), sysargs=["train", "--fake-data", "--n-fake-samples", "1"])
    assert output_model_path.exists()

    algo.execute(DummyAlgo(), sysargs=["train", "--debug"])
    assert output_model_path.exists()


def test_execute_train_multiple_models(workdir, create_models):
    _, model_filenames = create_models

    output_model_path = workdir / "model" / "model"
    assert not output_model_path.exists()
    pred_path = workdir / "pred" / "pred"
    assert not pred_path.exists()

    command = ["train"]
    command.extend(model_filenames)

    algo.execute(DummyAlgo(), sysargs=command)
    assert output_model_path.exists()
    with open(output_model_path, "r") as f:
        model = json.load(f)
    assert model["value"] == 3

    assert not pred_path.exists()


def test_execute_predict(workdir, create_models):
    _, model_filenames = create_models
    model_name = "model"
    output_model_path = workdir / "model" / model_name
    pred_path = workdir / "pred" / "pred"

    # first train models
    assert not pred_path.exists()
    command = ["train"]
    command.extend(model_filenames)
    algo.execute(DummyAlgo(), sysargs=command)
    assert output_model_path.exists()

    # do predict on output model
    assert not pred_path.exists()
    algo.execute(DummyAlgo(), sysargs=["predict", model_name])
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
    assert not pred_path.exists()
    algo.execute(
        DummyAlgo(),
        sysargs=["predict", "supermodel", "--models-path", str(input_models_dir)],
    )
    assert pred_path.exists()
    with open(pred_path, "r") as f:
        pred = json.load(f)
    assert pred == "XXX"


@pytest.mark.parametrize("algo_class", (NoSavedModelAlgo, WrongSavedModelAlgo))
def test_model_check(algo_class):
    a = algo_class()
    wp = algo.AlgoWrapper(a)

    with pytest.raises(exceptions.MissingFileError):
        wp.train([])


@pytest.mark.parametrize(
    "use_models_generator,models_type",
    (
        (True, types.GeneratorType),
        (False, list),
    ),
)
def test_models_generator(mocker, workdir, create_models, use_models_generator, models_type):
    _, model_filenames = create_models

    command = ["train"]
    command.extend(model_filenames)

    a = DummyAlgo()
    a.use_models_generator = use_models_generator
    mocker.patch.object(a, "train", autospec=True, return_value={})

    algo.execute(a, sysargs=command)
    models = a.train.call_args[0][2]
    assert isinstance(models, models_type)
