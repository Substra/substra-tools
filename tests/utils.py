import json
from os import PathLike
from typing import Any
from typing import List
from enum import Enum


class InputIdentifiers(str, Enum):
    local = "local"
    shared = "shared"
    model = "model"
    models = "models"
    predictions = "predictions"
    performance = "performance"
    opener = "opener"
    datasamples = "datasamples"
    X = "X"
    y = "y"
    rank = "rank"


class OutputIdentifiers(str, Enum):
    local = "local"
    shared = "shared"
    model = "model"
    predictions = "predictions"
    performance = "performance"


def load_models(paths: List[PathLike]) -> dict:
    models = []
    for model_path in paths:
        with open(model_path, "r") as f:
            models.append(json.load(f))

    return models


def load_model(path: PathLike):
    if path:
        with open(path, "r") as f:
            return json.load(f)


def save_model(model: dict, path: PathLike):
    with open(path, "w") as f:
        json.dump(model, f)


def save_predictions(predictions: Any, path: PathLike):
    with open(path, "w") as f:
        json.dump(predictions, f)


def load_predictions(path: PathLike) -> Any:
    with open(path, "r") as f:
        predictions = json.load(f)
    return predictions


def no_save_model(path, model):
    # do not save model at all
    pass


def wrong_save_model(model, path):
    # simulate numpy.save behavior
    with open(path + ".npy", "w") as f:
        json.dump(model, f)
