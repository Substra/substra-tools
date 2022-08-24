# coding: utf8
import abc
import argparse
import logging
import os
import sys
from enum import Enum
from typing import Any
from typing import List
from typing import Optional
from typing import TypedDict

from substratools import exceptions
from substratools import opener
from substratools import utils
from substratools.task_resources import COMPOSITE_IO_LOCAL
from substratools.task_resources import COMPOSITE_IO_SHARED
from substratools.task_resources import TASK_IO_CHAINKEYS
from substratools.task_resources import TASK_IO_DATASAMPLES
from substratools.task_resources import TASK_IO_LOCALFOLDER
from substratools.task_resources import TASK_IO_OPENER
from substratools.task_resources import TASK_IO_PREDICTIONS
from substratools.task_resources import TRAIN_IO_MODEL
from substratools.task_resources import TRAIN_IO_MODELS
from substratools.task_resources import TaskResources
from substratools.workspace import AggregateAlgoWorkspace
from substratools.workspace import AlgoWorkspace
from substratools.workspace import CompositeAlgoWorkspace

logger = logging.getLogger(__name__)

# TODO rework how to handle input args of command line commands and wrapper methods


class InputIdentifiers(str, Enum):
    local = "local"
    shared = "shared"
    model = "model"
    models = "models"
    predictions = "predictions"
    performance = "performance"
    opener = "opener"
    datasamples = "datasamples"
    rank = "rank"
    X = "X"
    y = "y"


class OutputIdentifiers(str, Enum):
    local = "local"
    shared = "shared"
    model = "model"
    predictions = "predictions"
    performance = "performance"


def _parser_add_default_arguments(parser):
    parser.add_argument(
        "-d",
        "--fake-data",
        action="store_true",
        default=False,
        help="Enable fake data mode",
    )
    parser.add_argument(
        "--n-fake-samples",
        default=None,
        type=int,
        help="Number of fake samples if fake data is used.",
    )
    parser.add_argument(
        "--log-path",
        default=None,
        help="Define log filename path",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=utils.MAPPING_LOG_LEVEL.keys(),
        help="Choose log level",
    )
    parser.add_argument(
        "--inputs",
        type=str,
        default="[]",
        help="Inputs of the compute task",
    )
    parser.add_argument(
        "--outputs",
        type=str,
        default="[]",
        help="Outputs of the compute task",
    )


class Algo(abc.ABC):
    """Abstract base class for defining algo to run on the platform.

    To define a new algo script, subclass this class and implement the
    following abstract methods:

    - #Algo.train()
    - #Algo.predict()

    The class has a `chainkeys_path` class property: it contains the path to the chainkeys folder.
    If the chainkey support is on, this folder contains the chainkeys.

    The class has a `compute_plan_path` class property: it contains the path to the compute plan local folder.
    If the algo is executed as part of a compute plan, this folder contains the shared data between the tasks of
    the compute plan.

    To add an algo to the Substra Platform, the line
    `tools.algo.execute(<AlgoClass>())` must be added to the main of the algo
    python script. It defines the algo command line interface and thus enables
    the Substra Platform to execute it.

    # Example

    ```python
    import json
    import substratools as tools


    class DummyAlgo(tools.Algo):
        def train(self, inputs, outputs):
            new_model = None
            self.save_model(new_model, outputs["model"])

        def predict(self, inputs, outputs):
            model = self.load_model(inputs["model"])
            predictions = 0
            self.save_predictions(predictions, outputs["predictions"])

        def load_model(self, path):
            return json.load(path)

        def save_model(self, model, path):
            json.dump(model, path)

        def save_predictions(self, predictions, path):
            json.dump(predictions, path)

    if __name__ == '__main__':
        tools.algo.execute(DummyAlgo())
    ```

    # How to test locally an algo script

    # Using the command line

    The algo script can be directly tested through it's command line interface.
    For instance to train an algo using fake data, run the following command:

    ```sh
    python <script_path> train --fake-data --n-fake-samples 20 --log-level debug
    ```

    To see all the available options for the train and predict commands, run:

    ```sh
    python <script_path> train --help
    python <script_path> predict --help
    ```

    # Using a python script

    An algo can be imported and used in python scripts as would any other class.

    For example, assuming that you have two local files named `opener.py` and
    `algo.py` (the latter containing an `Algo` class named `MyAlgo`):

    ```python
    import algo
    import opener

    o = opener.Opener()
    X = o.get_X(["dataset/train/train1"])
    y = o.get_y(["dataset/train/train1"])

    a = algo.MyAlgo()

    train_inputs={"X":X, "y":y, "model":None, "rank":0}
    train_outputs={"model":output_model_path}

    a.train(train_inputs, train_outputs)

    predict_inputs={"X":X, "model":input_model_path}
    predict_outputs={"predictions":output_predictions_path}

    a.predict(predict_inputs, predict_outputs)
    ```

    """

    chainkeys_path = None
    compute_plan_path = None

    @abc.abstractmethod
    def train(
        self,
        inputs: TypedDict(
            "inputs",
            {
                InputIdentifiers.X: List[Any],  # cf valid_opener_code # TODO: rename "data" , del Y ?
                InputIdentifiers.y: List[Any],  # datasamples contains loaded datasamples, if any, or None
                InputIdentifiers.models: Optional[
                    os.PathLike
                ],  # inputs contains a dict where keys are identifiers and values are paths on the disk
                InputIdentifiers.rank: int,
            },
        ),
        outputs: TypedDict(
            "outputs", {OutputIdentifiers.model: os.PathLike}
        ),  # outputs contains a dict where keys are identifiers and values are paths on disk
    ) -> None:
        """Train model and produce new model from train data.

        This task corresponds to the creation of a traintuple on the Substra
        Platform.

        # Arguments

        inputs: TypedDict(
            "inputs",
            {
                InputIdentifiers.X: List[Any]: training data samples loaded with `Opener.get_X()`.
                InputIdentifiers.y: List[Any]: training data samples labels loaded with `Opener.get_y()`.
                InputIdentifiers.models: Optional[
                    os.PathLike
                ]: list or generator of models loaded with `Algo.load_model()`.
                InputIdentifiers.rank: int: rank of the training task.
            },
        outputs: TypedDict(
            "outputs", {OutputIdentifiers.model: os.PathLike}: output model path to save the model.
        )
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(
        self,
        inputs: TypedDict(
            "inputs",
            {
                InputIdentifiers.X: Any,
                InputIdentifiers.model: List[os.PathLike],
            },
        ),
        outputs: TypedDict(
            "outputs",
            {
                OutputIdentifiers.predictions: os.PathLike,
            },
        ),
    ) -> None:
        """Get predictions from test data.

        This task corresponds to the creation of a testtuple on the Substra
        Platform.

        # Arguments

        inputs: TypedDict(
            "inputs",
            {
                InputIdentifiers.X: Any: testing data samples loaded with `Opener.get_X()`.
                InputIdentifiers.model: List[os.PathLike]: input model load with `Algo.load_model()` used for
                    predictions.
            },
        ),
        outputs: TypedDict(
            "outputs",
            {
                OutputIdentifiers.predictions: os.PathLike: output predictions path to save the predictions.
            },
        )
        """
        raise NotImplementedError

    def _train_fake_data(self, *args, **kwargs):
        """Train model fake data mode.

        This method is called by the algorithm wrapper when the fake data mode
        is enabled. In fake data mode, `X` and `y` input args have been
        replaced by the opener fake data.

        By default, it only calls directly `Algo.train()` method. Override this
        method if you want to implement a different behavior.
        """
        return self.train(*args, **kwargs)

    def _predict_fake_data(self, *args, **kwargs):
        """Predict model fake data mode.

        This method is called by the algorithm wrapper when the fake data mode
        is enabled. In fake data mode, `X` input arg has been replaced by
        the opener fake data.

        By default, it only calls directly `Algo.predict()` method. Override
        this method if you want to implement a different behavior.
        """
        return self.predict(*args, **kwargs)


class AlgoWrapper(object):
    """Algo wrapper to execute an algo instance on the platform."""

    _INTERFACE_CLASS = Algo

    def __init__(self, interface, workspace, opener_wrapper=None):
        assert isinstance(interface, self._INTERFACE_CLASS)
        self._workspace = workspace
        self._opener_wrapper = opener_wrapper or opener.load_from_module(workspace=self._workspace)
        self._interface = interface

        self._interface.chainkeys_path = self._workspace.chainkeys_path
        self._interface.compute_plan_path = self._workspace.compute_plan_path

    def _assert_output_exists(self, path, key):

        if os.path.isdir(path):
            raise exceptions.NotAFileError(f"Expected output file at {path}, found dir for output `{key}`")
        if not os.path.isfile(path):
            raise exceptions.MissingFileError(f"Output file {path} used to save argument `{key}` does not exists.")

    def _load_models_paths(self):
        if not self._workspace.input_model_paths:
            return []
        return self._workspace.input_model_paths

    @utils.Timer(logger)
    def train(self, rank=0, fake_data=False, n_fake_samples=None):
        """Train method wrapper."""
        # load data from opener
        X = self._opener_wrapper.get_X(fake_data, n_fake_samples)
        y = self._opener_wrapper.get_y(fake_data, n_fake_samples)

        # load models
        models = self._load_models_paths()

        # train new model
        logger.info("launching training task")
        method = self._interface.train if not fake_data else self._interface._train_fake_data

        # TODO get data and models labels from --inputs
        method(
            inputs={
                InputIdentifiers.X: X,
                InputIdentifiers.y: y,
                InputIdentifiers.models: models,
                InputIdentifiers.rank: rank,
            },
            outputs={OutputIdentifiers.model: self._workspace.output_model_path},
        )

        self._assert_output_exists(self._workspace.output_model_path, OutputIdentifiers.model)

    @utils.Timer(logger)
    def predict(self, fake_data=False, n_fake_samples=None):
        """Predict method wrapper."""
        # load data from opener
        X = self._opener_wrapper.get_X(fake_data, n_fake_samples)

        # load models
        model_paths = self._workspace.input_model_paths
        if not model_paths or len(model_paths) != 1:
            raise exceptions.InvalidInputOutputsError("predict expects exactly one input model")
        model_path = model_paths[0]

        # get predictions
        logger.info("launching predict task")
        method = self._interface.predict if not fake_data else self._interface._predict_fake_data
        inputs = {InputIdentifiers.X: X, InputIdentifiers.model: model_path}
        outputs = {OutputIdentifiers.predictions: self._workspace.output_predictions_path}

        method(inputs, outputs)

        self._assert_output_exists(self._workspace.output_predictions_path, OutputIdentifiers.predictions)


def _generate_algo_cli(interface):
    """Helper to generate a command line interface client."""

    def _algo_from_args(args):
        inputs = TaskResources(args.inputs)
        outputs = TaskResources(args.outputs)
        workspace = AlgoWorkspace(
            # Data samples are optional because we could be using fake data.
            # TODO: validate that the input contains either the --fake-data argument OR a "datasample" input.
            input_data_folder_paths=inputs.get_optional_values(TASK_IO_DATASAMPLES),
            log_path=args.log_path,
            input_model_paths=inputs.get_optional_values(TRAIN_IO_MODELS),
            output_model_path=outputs.get_optional_value(TRAIN_IO_MODEL),
            output_predictions_path=outputs.get_optional_value(TASK_IO_PREDICTIONS),
            chainkeys_path=inputs.get_optional_value(TASK_IO_CHAINKEYS),
            compute_plan_path=inputs.get_optional_value(TASK_IO_LOCALFOLDER),
        )
        utils.configure_logging(workspace.log_path, log_level=args.log_level)
        opener_wrapper = opener.load_from_module(
            path=inputs.get_optional_value(TASK_IO_OPENER),
            workspace=workspace,
        )
        return AlgoWrapper(interface, workspace, opener_wrapper)

    def _train(args):
        algo_wrapper = _algo_from_args(args)
        algo_wrapper.train(args.rank, args.fake_data, args.n_fake_samples)

    parser = argparse.ArgumentParser()
    parsers = parser.add_subparsers()
    train_parser = parsers.add_parser("train")
    train_parser.add_argument(
        "-r",
        "--rank",
        type=int,
        default=0,
        help="Define machine learning task rank",
    )
    _parser_add_default_arguments(train_parser)
    train_parser.set_defaults(func=_train)

    def _predict(args):
        algo_wrapper = _algo_from_args(args)
        algo_wrapper.predict(args.fake_data, args.n_fake_samples)

    predict_parser = parsers.add_parser("predict")
    _parser_add_default_arguments(predict_parser)
    predict_parser.set_defaults(func=_predict)

    return parser


class CompositeAlgo(abc.ABC):
    """Abstract base class for defining a composite algo to run on the platform.

    To define a new composite algo script, subclass this class and implement the
    following abstract methods:

    - #CompositeAlgo.train()
    - #CompositeAlgo.predict()

    To add a composite algo to the Substra Platform, the line
    `tools.algo.execute(<CompositeAlgoClass>())` must be added to the main of the algo
    python script. It defines the composite algo command line interface and thus enables
    the Substra Platform to execute it.

    The class has a `chainkeys_path` class property: it contains the path to the chainkeys folder.
    If the chainkey support is on, this folder contains the chainkeys.

    The class has a `compute_plan_path` class property: it contains the path to the compute plan local folder.
    If the algo is executed as part of a compute plan, this folder contains the shared data between the tasks of
    the compute plan.

    # Example

    ```python
    import json
    import substratools as tools


    class DummyCompositeAlgo(tools.CompositeAlgo):
        def train(self, inputs, outputs):
            new_head_model = None
            new_trunk_model = None
            self.save_model(new_head_model, outputs["local"])
            self.save_model(new_trunk_model, outputs["shared"])

        def predict(self, inputs, outputs):
            predictions = 0
            self.save_predictions(predictions, outputs["predictions”])

        def load_model(self, path):
            return json.load(path)

        def save_model(self, model, path):
            json.dump(model, path)

        def save_predictions(self, predictions, path):
            json.dump(predictions, path)


    if __name__ == '__main__':
        tools.algo.execute(DummyCompositeAlgo())
    ```

    # How to test locally a composite algo script

    # Using the command line

    The composite algo script can be directly tested through it's command line interface.
    For instance to train an algo using fake data, run the following command:

    ```sh
    python <script_path> train --fake-data --n-fake-samples 20 --log-level debug
    ```

    To see all the available options for the train and predict commands, run:

    ```sh
    python <script_path> train --help
    python <script_path> predict --help
    ```

    # Using a python script

    A composite algo can be imported and used in python scripts as would any other class.

    For example, assuming that you have two local files named `opener.py` and
    `composite_algo.py` (the latter containing a `CompositeAlgo` class named
    `MyCompositeAlgo`):

    ```python
    import composite_algo
    import opener

    o = opener.Opener()
    X = o.get_X(["dataset/train/train1"])
    y = o.get_y(["dataset/train/train1"])

    a = composite_algo.MyCompositeAlgo()
    inputs_train = {"X":X, "y":y, "local":None, "shared":None, "rank":0}
    outputs_train = {"local":head_model_path, "shared":trunk_model_path}
    head_model, trunk_model = a.train(inputs_train, outputs_train)

    inputs_predict = {"X":X, "local":None, "shared":None}
    outputs_predict = {"predictions":predictions_path}
    y_pred = a.predict(inputs_predict, outputs_predict)
    ```
    """

    chainkeys_path = None
    compute_plan_path = None

    @abc.abstractmethod
    def train(
        self,
        inputs: TypedDict(
            "inputs",
            {
                InputIdentifiers.X: Any,
                InputIdentifiers.y: Any,
                InputIdentifiers.local: Optional[os.PathLike],
                InputIdentifiers.shared: Optional[os.PathLike],
                InputIdentifiers.rank: int,
            },
        ),
        outputs: TypedDict(
            "outputs",
            {
                OutputIdentifiers.local: os.PathLike,
                OutputIdentifiers.shared: os.PathLike,
            },
        ),  # outputs contains a dict where keys are identifiers and values are paths on disk
    ) -> None:
        """Train model and produce new composite models from train data.

        This task corresponds to the creation of a composite traintuple on the Substra
        Platform.

        # Arguments

        inputs: TypedDict(
            "inputs",
            {
                InputIdentifiers.X: Any: training data samples loaded with `Opener.get_X()`.
                InputIdentifiers.y: Any: training data samples labels loaded with `Opener.get_y()`.
                InputIdentifiers.local: Optional[os.PathLike]: head model loaded with `CompositeAlgo.load_head_model()`
                    (may be None).
                InputIdentifiers.shared: Optional[os.PathLike]: trunk model loaded with
                    `CompositeAlgo.load_trunk_model()` (may be None).
                InputIdentifiers.rank: int: rank of the training task.
            },
        ),
        outputs: TypedDict(
            "outputs",
            {
                OutputIdentifiers.local: os.PathLike: output head model path to save the head model.
                OutputIdentifiers.shared: os.PathLike: output trunk model path to save the trunk model.
            }
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(
        self,
        inputs: TypedDict(
            "inputs",
            {
                InputIdentifiers.X: Any,
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
    ) -> None:
        """Get predictions from test data.

        This task corresponds to the creation of a composite testtuple on the Substra
        Platform.

        # Arguments

        inputs: TypedDict(
            "inputs",
            {
                InputIdentifiers.X: Any: testing data samples loaded with `Opener.get_X()`.
                InputIdentifiers.local: os.PathLike: head model loaded with `CompositeAlgo.load_head_model()`.
                InputIdentifiers.shared: os.PathLike: trunk model loaded with `CompositeAlgo.load_trunk_model()`.
            },
        ),
        outputs: TypedDict(
            "outputs",
            {
                OutputIdentifiers.predictions: os.PathLike: output predictions path to save the predictions.
            },
        )
        """
        raise NotImplementedError

    def _train_fake_data(self, *args, **kwargs):
        """Train model fake data mode.

        This method is called by the algorithm wrapper when the fake data mode
        is enabled. In fake data mode, `X` and `y` input args have been
        replaced by the opener fake data.

        By default, it only calls directly `Algo.train()` method. Override this
        method if you want to implement a different behavior.
        """
        return self.train(*args, **kwargs)

    def _predict_fake_data(self, *args, **kwargs):
        """Predict model fake data mode.

        This method is called by the algorithm wrapper when the fake data mode
        is enabled. In fake data mode, `X` input arg has been replaced by
        the opener fake data.

        By default, it only calls directly `Algo.predict()` method. Override
        this method if you want to implement a different behavior.
        """
        return self.predict(*args, **kwargs)


class CompositeAlgoWrapper(AlgoWrapper):
    """Algo wrapper to execute an algo instance on the platform."""

    _INTERFACE_CLASS = CompositeAlgo

    def _load_models_paths(self):
        """Load head and trunk models from their filename."""
        head_model_path = None
        if self._workspace.input_head_model_path:
            head_model_path = self._workspace.input_head_model_path
        trunk_model_path = None
        if self._workspace.input_trunk_model_path:
            trunk_model_path = self._workspace.input_trunk_model_path
        return head_model_path, trunk_model_path

    @utils.Timer(logger)
    def train(
        self,
        rank=0,
        fake_data=False,
        n_fake_samples=None,
    ):
        """Train method wrapper."""
        # load data from opener
        X = self._opener_wrapper.get_X(fake_data, n_fake_samples)
        y = self._opener_wrapper.get_y(fake_data, n_fake_samples)

        # load head and trunk models
        head_model_path, trunk_model_path = self._load_models_paths()

        # train new models
        logger.info("launching training task")
        method = self._interface.train if not fake_data else self._interface._train_fake_data

        method(
            inputs={
                InputIdentifiers.X: X,
                InputIdentifiers.y: y,
                InputIdentifiers.shared: trunk_model_path,
                InputIdentifiers.local: head_model_path,
                InputIdentifiers.rank: rank,
            },
            outputs={
                OutputIdentifiers.local: self._workspace.output_head_model_path,
                OutputIdentifiers.shared: self._workspace.output_trunk_model_path,
            },
        )

        self._assert_output_exists(self._workspace.output_head_model_path, OutputIdentifiers.local)
        self._assert_output_exists(self._workspace.output_trunk_model_path, OutputIdentifiers.shared)

    @utils.Timer(logger)
    def predict(self, fake_data=False, n_fake_samples=None):
        """Predict method wrapper."""
        # load data from opener
        X = self._opener_wrapper.get_X(fake_data, n_fake_samples)

        # load head and trunk models
        head_model_path, trunk_model_path = self._load_models_paths()
        assert head_model_path and trunk_model_path  # should not be None

        # get predictions
        logger.info("launching predict taske")
        method = self._interface.predict if not fake_data else self._interface._predict_fake_data

        inputs = {
            InputIdentifiers.X: X,
            InputIdentifiers.local: head_model_path,
            InputIdentifiers.shared: trunk_model_path,
        }

        outputs = {OutputIdentifiers.predictions: self._workspace.output_predictions_path}

        method(inputs, outputs)

        self._assert_output_exists(self._workspace.output_predictions_path, OutputIdentifiers.predictions)


def _generate_composite_algo_cli(interface):
    """Helper to generate a command line interface client."""

    def _algo_from_args(args):
        inputs = TaskResources(args.inputs)
        outputs = TaskResources(args.outputs)
        workspace = CompositeAlgoWorkspace(
            # Data samples are optional because we could be using fake data.
            # TODO: validate that the input contains either the --fake-data argument OR a "datasample" input.
            input_data_folder_paths=inputs.get_optional_values(TASK_IO_DATASAMPLES),
            log_path=args.log_path,
            chainkeys_path=inputs.get_optional_value(TASK_IO_CHAINKEYS),
            compute_plan_path=inputs.get_optional_value(TASK_IO_LOCALFOLDER),
            input_head_model_path=inputs.get_optional_value(COMPOSITE_IO_LOCAL),
            input_trunk_model_path=inputs.get_optional_value(COMPOSITE_IO_SHARED),
            output_head_model_path=outputs.get_optional_value(COMPOSITE_IO_LOCAL),
            output_trunk_model_path=outputs.get_optional_value(COMPOSITE_IO_SHARED),
            output_predictions_path=outputs.get_optional_value(TASK_IO_PREDICTIONS),
        )
        opener_wrapper = opener.load_from_module(
            path=inputs.get_optional_value(TASK_IO_OPENER),
            workspace=workspace,
        )
        utils.configure_logging(workspace.log_path, log_level=args.log_level)
        return CompositeAlgoWrapper(interface, workspace, opener_wrapper)

    def _train(args):
        algo_wrapper = _algo_from_args(args)
        algo_wrapper.train(
            args.rank,
            args.fake_data,
            args.n_fake_samples,
        )

    parser = argparse.ArgumentParser()
    parsers = parser.add_subparsers()
    train_parser = parsers.add_parser("train")
    train_parser.add_argument(
        "-r",
        "--rank",
        type=int,
        default=0,
        help="Define machine learning task rank",
    )
    _parser_add_default_arguments(train_parser)
    train_parser.set_defaults(func=_train)

    def _predict(args):
        algo_wrapper = _algo_from_args(args)
        algo_wrapper.predict(args.fake_data, args.n_fake_samples)

    predict_parser = parsers.add_parser("predict")
    _parser_add_default_arguments(predict_parser)
    predict_parser.set_defaults(func=_predict)

    return parser


class AggregateAlgo(abc.ABC):
    """Abstract base class for defining an aggregate algo to run on the platform.

    To define a new aggregate algo script, subclass this class and implement the
    following abstract methods:

    - #AggregateAlgo.aggregate()
    - #AggregateAlgo.predict()

    The class has a `chainkeys_path` class property: it contains the path to the chainkeys folder.
    If the chainkey support is on, this folder contains the chainkeys.

    The class has a `compute_plan_path` class property: it contains the path to the compute plan local folder.
    If the algo is executed as part of a compute plan, this folder contains the shared data between the tasks of
    the compute plan.

    To add a aggregate algo to the Substra Platform, the line
    `tools.algo.execute(<AggregateAlgoClass>())` must be added to the main of the algo
    python script. It defines the aggregate algo command line interface and thus enables
    the Substra Platform to execute it.

    # Example

    ```python
    import json
    import substratools as tools


    class DummyAggregateAlgo(tools.AggregateAlgo):
        def aggregate(self, inputs, outputs):
            new_model = None
            self.save_model(outputs["model"])

        def predict(self, inputs, outputs):
            predictions = 0
            self.save_predictions(predictions, outputs["predictions”])

        def load_model(self, path):
            return json.load(path)

        def save_model(self, model, path):
            json.dump(model, path)

        def save_predictions(self, predictions, path):
            json.dump(predictions, path)


    if __name__ == '__main__':
        tools.algo.execute(DummyAggregateAlgo())
    ```

    # How to test locally an aggregate algo script

    # Using the command line

    The aggregate algo script can be directly tested through it's command line interface.
    For instance to train an algo using fake data, run the following command:

    ```sh
    python <script_path> aggregate --models_path <models_path> --models <model_name> --model <model_name> \
    --log-level debug
    ```

    To see all the available options for the aggregate and predict commands, run:

    ```sh
    python <script_path> aggregate --help
    python <script_path> predict --help
    ```

    # Using a python script

    An aggregate algo can be imported and used in python scripts as would any other class.

    For example, assuming that you have a local file named `aggregate_algo.py` containing
    containing an `AggregateAlgo` class named `MyAggregateAlgo`:

    ```python
    from aggregate_algo import MyAggregateAlgo

    a = MyAggregateAlgo()

    model_1 = a.load_model('./sandbox/models/model_1')
    model_2 = a.load_model('./sandbox/models/model_2')

    aggregated_model = a.aggregate([model_1, model_2], 0)
    ```
    """

    chainkeys_path = None
    compute_plan_path = None

    @abc.abstractmethod
    def aggregate(
        self,
        inputs: TypedDict(
            "inputs",
            {
                InputIdentifiers.models: List[os.PathLike],
                InputIdentifiers.rank: int,
            },
        ),
        outputs: TypedDict("outputs", {OutputIdentifiers.model: os.PathLike}),
    ):
        """Aggregate models and produce a new model.

        This task corresponds to the creation of an aggregate tuple on the Substra
        Platform.

        # Arguments

        inputs: TypedDict(
            "inputs",
            {
                InputIdentifiers.models: List[os.PathLike]: list of models path loaded with `AggregateAlgo.load_model()`
                InputIdentifiers.rank: int: rank of the aggregate task.
            },
        ),
        outputs: TypedDict("outputs", {OutputIdentifiers.model: os.PathLike}): output model path to save the aggregated
            model.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(
        self,
        inputs: TypedDict(
            "inputs",
            {
                InputIdentifiers.X: Any,
                InputIdentifiers.model: os.PathLike,
            },
        ),
        outputs: TypedDict("outputs", {"model": os.PathLike}),
    ):
        """Get predictions from test data.

        This task corresponds to the creation of a testtuple on the Substra
        Platform.

        # Arguments

        inputs: TypedDict(
            "inputs",
            {
                InputIdentifiers.X: Any: testing data samples loaded with `Opener.get_X()`.
                InputIdentifiers.model: os.PathLike: input model load with `AggregateAlgo.load_model()` used for
                predictions.
            },
        ),
        outputs: TypedDict("outputs", {"model": os.PathLike}): output predictions path to save the predictions.
        """
        raise NotImplementedError

    def _predict_fake_data(self, *args, **kwargs):
        """Predict model fake data mode.

        This method is called by the algorithm wrapper when the fake data mode
        is enabled. In fake data mode, `X` input arg has been replaced by
        the opener fake data.

        By default, it only calls directly `Algo.predict()` method. Override
        this method if you want to implement a different behavior.
        """
        return self.predict(*args, **kwargs)


class AggregateAlgoWrapper(object):
    """Aggregate algo wrapper to execute an aggregate algo instance on the platform."""

    def __init__(self, interface, workspace, opener_wrapper=None):
        assert isinstance(interface, AggregateAlgo)
        self._workspace = workspace
        self._opener_wrapper = opener_wrapper
        self._interface = interface

        self._interface.chainkeys_path = self._workspace.chainkeys_path
        self._interface.compute_plan_path = self._workspace.compute_plan_path

    def _get_models_paths(self):
        if not self._workspace.input_model_paths:
            return []
        return self._workspace.input_model_paths

    def _assert_output_exists(self, path, key):

        if os.path.isdir(path):
            raise exceptions.NotAFileError(f"Expected output file at {path}, found dir for output `{key}`")
        if not os.path.isfile(path):
            raise exceptions.MissingFileError(f"Output file {path} used to save argument `{key}` does not exists.")

    @utils.Timer(logger)
    def aggregate(self, rank=0):
        """Aggregate method wrapper."""
        # load models
        models = self._get_models_paths()

        # train new model
        logger.info("launching aggregate task")

        inputs = {InputIdentifiers.models: models, InputIdentifiers.rank: rank}
        outputs = {OutputIdentifiers.model: self._workspace.output_model_path}

        self._interface.aggregate(inputs, outputs)

        # serialize output model and save it to workspace
        logger.info("saving output model to '{}'".format(self._workspace.output_model_path))
        self._assert_output_exists(self._workspace.output_model_path, OutputIdentifiers.model)

    @utils.Timer(logger)
    def predict(self, fake_data=False, n_fake_samples=None):
        """Predict method wrapper."""
        # lazy load of opener wrapper as it is required only for the predict
        self._opener_wrapper = self._opener_wrapper or opener.load_from_module(workspace=self._workspace)
        # load data from opener
        X = self._opener_wrapper.get_X(fake_data, n_fake_samples)

        # load models
        model_paths = self._workspace.input_model_paths
        if len(model_paths) != 1:
            raise exceptions.InvalidInputOutputsError("predict expects exactly one input model")
        model_path = self._get_models_paths()[0]

        # get predictions
        logger.info("launching predict task")
        method = self._interface.predict if not fake_data else self._interface._predict_fake_data

        inputs = {InputIdentifiers.X: X, InputIdentifiers.model: model_path}
        outputs = {OutputIdentifiers.predictions: self._workspace.output_predictions_path}

        method(inputs, outputs)

        self._assert_output_exists(self._workspace.output_predictions_path, OutputIdentifiers.predictions)


def _generate_aggregate_algo_cli(interface):
    """Helper to generate a command line interface client."""

    def _algo_from_args(args):
        inputs = TaskResources(args.inputs)
        outputs = TaskResources(args.outputs)
        workspace = AggregateAlgoWorkspace(
            input_data_folder_paths=inputs.get_optional_values(TASK_IO_DATASAMPLES),
            input_model_paths=inputs.get_optional_values(TRAIN_IO_MODELS),
            log_path=args.log_path,
            output_model_path=outputs.get_optional_value(TRAIN_IO_MODEL),
            output_predictions_path=outputs.get_optional_value(TASK_IO_PREDICTIONS),
            chainkeys_path=inputs.get_optional_value(TASK_IO_CHAINKEYS),
            compute_plan_path=inputs.get_optional_value(TASK_IO_LOCALFOLDER),
        )
        utils.configure_logging(workspace.log_path, log_level=args.log_level)
        if inputs.get_optional_value(TASK_IO_OPENER):
            opener_wrapper = opener.load_from_module(
                path=inputs.get_optional_value(TASK_IO_OPENER),
                workspace=workspace,
            )
        else:
            opener_wrapper = None
        return AggregateAlgoWrapper(interface, workspace, opener_wrapper)

    def _aggregate(args):
        algo_wrapper = _algo_from_args(args)
        algo_wrapper.aggregate(args.rank)

    parser = argparse.ArgumentParser()
    parsers = parser.add_subparsers()
    aggregate_parser = parsers.add_parser("aggregate")
    aggregate_parser.add_argument(
        "-r",
        "--rank",
        type=int,
        default=0,
        help="Define machine learning task rank",
    )
    _parser_add_default_arguments(aggregate_parser)
    aggregate_parser.set_defaults(func=_aggregate)

    def _predict(args):
        algo_wrapper = _algo_from_args(args)
        algo_wrapper.predict(args.fake_data, args.n_fake_samples)

    predict_parser = parsers.add_parser("predict")
    _parser_add_default_arguments(predict_parser)
    predict_parser.set_defaults(func=_predict)

    return parser


def execute(interface, sysargs=None):
    """Launch algo command line interface."""
    if isinstance(interface, AggregateAlgo):
        generator = _generate_aggregate_algo_cli
    elif isinstance(interface, CompositeAlgo):
        generator = _generate_composite_algo_cli
    else:
        generator = _generate_algo_cli

    cli = generator(interface)

    sysargs = sysargs if sysargs is not None else sys.argv[1:]
    args = cli.parse_args(sysargs)
    args.func(args)
    return args
