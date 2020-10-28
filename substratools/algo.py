# coding: utf8
import abc
import argparse
import logging
import os
import sys

from substratools import opener, utils, exceptions
from substratools.workspace import (AlgoWorkspace, CompositeAlgoWorkspace,
                                    AggregateAlgoWorkspace)


logger = logging.getLogger(__name__)

# TODO rework how to handle input args of command line commands and wrapper methods


class Algo(abc.ABC):
    """Abstract base class for defining algo to run on the platform.

    To define a new algo script, subclass this class and implement the
    following abstract methods:

    - #Algo.train()
    - #Algo.predict()
    - #Algo.load_model()
    - #Algo.save_model()

    This class has an `use_models_generator` class property:
    - if True, models will be passed to the `train` method as a generator
    - (default) if False, models will be passed to the `train` method as a list

    To add an algo to the Substra Platform, the line
    `tools.algo.execute(<AlgoClass>())` must be added to the main of the algo
    python script. It defines the algo command line interface and thus enables
    the Substra Platform to execute it.

    # Example

    ```python
    import json
    import substratools as tools


    class DummyAlgo(tools.Algo):
        def train(self, X, y, models, rank):
            new_model = None
            return new_model

        def predict(self, X, model):
            predictions = 0
            return predictions

        def load_model(self, path):
            return json.load(path)

        def save_model(self, model, path):
            json.dump(model, path)


    if __name__ == '__main__':
        tools.algo.execute(DummyAlgo())
    ```

    # How to test locally an algo script

    # Using the command line

    The algo script can be directly tested through it's command line interface.
    For instance to train an algo using fake data, run the following command:

    ```sh
    python <script_path> train --fake-data --n-fake-samples 20 --debug
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
    model = a.train(X, y, None, None, 0)
    y_pred = a.predict(X, model)
    ```

    """

    use_models_generator = False

    @abc.abstractmethod
    def train(self, X, y, models, rank):
        """Train model and produce new model from train data.

        This task corresponds to the creation of a traintuple on the Substra
        Platform.

        # Arguments

        X: training data samples loaded with `Opener.get_X()`.
        y: training data samples labels loaded with `Opener.get_y()`.
        models: list or generator of models loaded with `Algo.load_model()`.
        rank: rank of the training task.

        # Returns

        model: model object.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, X, model):
        """Get predictions from test data.

        This task corresponds to the creation of a testtuple on the Substra
        Platform.

        # Arguments

        X: testing data samples loaded with `Opener.get_X()`.
        model: input model load with `Algo.load_model()` used for predictions.

        # Returns

        predictions: predictions object.
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

    @abc.abstractmethod
    def load_model(self, path):
        """Deserialize model from file.

        This method will be executed before the call to the methods
        `Algo.train()` and `Algo.predict()` to deserialize the model objects.

        # Arguments

        path: path of the model to load.

        # Returns

        model: the deserialized model object.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def save_model(self, model, path):
        """Serialize model in file.

        This method will be executed after the call to the methods
        `Algo.train()` and `Algo.predict()` to save the model objects.

        # Arguments

        path: path of file to write.
        model: the model to serialize.
        """
        raise NotImplementedError


class AlgoWrapper(object):
    """Algo wrapper to execute an algo instance on the platform."""
    _INTERFACE_CLASS = Algo
    _DEFAULT_WORKSPACE_CLASS = AlgoWorkspace

    def __init__(self, interface, workspace=None, opener_wrapper=None):
        assert isinstance(interface, self._INTERFACE_CLASS)
        self._workspace = workspace or self._DEFAULT_WORKSPACE_CLASS()
        self._opener_wrapper = opener_wrapper or \
            opener.load_from_module(workspace=self._workspace)
        self._interface = interface

    def _assert_output_model_exists(self):
        path = self._workspace.output_model_path
        if os.path.isdir(path):
            raise exceptions.NotAFileError(f'Expected output model file at {path}, found dir')
        if not os.path.isfile(path):
            raise exceptions.MissingFileError(f'Output model file {path} does not exists')

    def _load_model(self, model_name):
        """Load single model in memory from its name."""
        # load model from workspace and deserialize it
        model_path = os.path.join(self._workspace.input_models_folder_path, model_name)
        logger.info("loading model from '{}'".format(model_path))
        return self._interface.load_model(model_path)

    def _load_models_as_list(self, model_names):
        return [self._load_model(model_name) for model_name in model_names]

    def _load_models_as_generator(self, model_names):
        for model_name in model_names:
            yield self._load_model(model_name)

    def _load_models(self, model_names):
        """Load models either as list or as generator"""
        if self._interface.use_models_generator:
            return self._load_models_as_generator(model_names)
        return self._load_models_as_list(model_names)

    def train(self, model_names, rank=0, fake_data=False, n_fake_samples=None):
        """Train method wrapper."""
        # load data from opener
        X = self._opener_wrapper.get_X(fake_data, n_fake_samples)
        y = self._opener_wrapper.get_y(fake_data, n_fake_samples)

        # load models
        models = self._load_models(model_names)

        # train new model
        logger.info("launching training task")
        method = (self._interface.train if not fake_data else
                  self._interface._train_fake_data)
        model = method(X, y, models, rank)

        # serialize output model and save it to workspace
        logger.info("saving output model to '{}'".format(
            self._workspace.output_model_path))
        self._interface.save_model(model, self._workspace.output_model_path)
        self._assert_output_model_exists()

        return model

    def predict(self, model_name, fake_data=False, n_fake_samples=None):
        """Predict method wrapper."""
        # load data from opener
        X = self._opener_wrapper.get_X(fake_data, n_fake_samples)

        # load models
        model = self._load_model(model_name)

        # get predictions
        logger.info("launching predict task")
        method = (self._interface.predict if not fake_data else
                  self._interface._predict_fake_data)
        pred = method(X, model)

        # save predictions
        self._opener_wrapper.save_predictions(pred)
        return pred


def _generate_algo_cli(interface):
    """Helper to generate a command line interface client."""

    def _algo_from_args(args):
        workspace = AlgoWorkspace(
            input_data_folder_paths=args.data_sample_paths,
            input_models_folder_path=args.models_path,
            log_path=args.log_path,
            output_model_path=args.output_model_path,
            output_predictions_path=args.output_predictions_path,
        )
        utils.configure_logging(workspace.log_path, debug_mode=args.debug)
        opener_wrapper = opener.load_from_module(
            path=args.opener_path,
            workspace=workspace,
        )
        return AlgoWrapper(
            interface,
            workspace=workspace,
            opener_wrapper=opener_wrapper,
        )

    def _parser_add_default_arguments(_parser):
        _parser.add_argument(
            '-d', '--fake-data', action='store_true', default=False,
            help="Enable fake data mode",
        )
        _parser.add_argument(
            '--n-fake-samples', default=None, type=int,
            help="Number of fake samples if fake data is used.",
        )
        _parser.add_argument(
            '--data-sample-paths', default=[],
            nargs='*',
            help="Define train/test data samples folder paths",
        )
        _parser.add_argument(
            '--models-path', default=None,
            help="Define models folder path",
        )
        _parser.add_argument(
            '--output-model-path', default=None,
            help="Define output model file path",
        )
        _parser.add_argument(
            '--output-predictions-path', default=None,
            help="Define output predictions file path",
        )
        _parser.add_argument(
            '--log-path', default=None,
            help="Define log filename path",
        )
        _parser.add_argument(
            '--opener-path', default=None,
            help="Define path to opener python script",
        )
        _parser.add_argument(
            '--debug', action='store_true', default=False,
            help="Enable debug mode (logs printed in stdout)",
        )

    def _train(args):
        algo_wrapper = _algo_from_args(args)
        algo_wrapper.train(
            args.models,
            args.rank,
            args.fake_data,
            args.n_fake_samples
        )

    parser = argparse.ArgumentParser()
    parsers = parser.add_subparsers()
    train_parser = parsers.add_parser('train')
    train_parser.add_argument(
        'models', type=str, nargs='*',
        help="Model names (must be located in default models folder)"
    )
    train_parser.add_argument(
        '-r', '--rank', type=int, default=0,
        help="Define machine learning task rank",
    )
    _parser_add_default_arguments(train_parser)
    train_parser.set_defaults(func=_train)

    def _predict(args):
        algo_wrapper = _algo_from_args(args)
        algo_wrapper.predict(
            args.model,
            args.fake_data,
            args.n_fake_samples
        )

    predict_parser = parsers.add_parser('predict')
    predict_parser.add_argument(
        'model', type=str,
        help="Model name (must be located in default models folder)"
    )
    _parser_add_default_arguments(predict_parser)
    predict_parser.set_defaults(func=_predict)

    return parser


class CompositeAlgo(abc.ABC):
    """Abstract base class for defining a composite algo to run on the platform.

    To define a new composite algo script, subclass this class and implement the
    following abstract methods:

    - #CompositeAlgo.train()
    - #CompositeAlgo.predict()
    - #CompositeAlgo.load_head_model()
    - #CompositeAlgo.save_head_model()
    - #CompositeAlgo.load_trunk_model()
    - #CompositeAlgo.save_trunk_model()

    To add a composite algo to the Substra Platform, the line
    `tools.algo.execute(<CompositeAlgoClass>())` must be added to the main of the algo
    python script. It defines the composite algo command line interface and thus enables
    the Substra Platform to execute it.

    # Example

    ```python
    import json
    import substratools as tools


    class DummyCompositeAlgo(tools.CompositeAlgo):
        def train(self, X, y, head_model, trunk_model, rank):
            new_head_model = None
            new_trunk_model = None
            return new_head_model, new_trunk_model

        def predict(self, X, head_model, trunk_model):
            predictions = 0
            return predictions

        def load_head_model(self, path):
            return json.load(path)

        def save_head_model(self, model, path):
            json.dump(model, path)

        def load_trunk_model(self, path):
            return json.load(path)

        def save_trunk_model(self, model, path):
            json.dump(model, path)


    if __name__ == '__main__':
        tools.algo.execute(DummyCompositeAlgo())
    ```

    # How to test locally a composite algo script

    # Using the command line

    The composite algo script can be directly tested through it's command line interface.
    For instance to train an algo using fake data, run the following command:

    ```sh
    python <script_path> train --fake-data --n-fake-samples 20 --debug
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
    head_model, trunk_model = a.train(X, y, None, None, 0)
    y_pred = a.predict(X, head_model, trunk_model)
    ```
    """

    @abc.abstractmethod
    def train(self, X, y, head_model, trunk_model, rank):
        """Train model and produce new composite models from train data.

        This task corresponds to the creation of a composite traintuple on the Substra
        Platform.

        # Arguments

        X: training data samples loaded with `Opener.get_X()`.
        y: training data samples labels loaded with `Opener.get_y()`.
        head_model: head model loaded with `CompositeAlgo.load_head_model()` (may be None).
        trunk_model: trunk model loaded with `CompositeAlgo.load_trunk_model()` (may be None).
        rank: rank of the training task.

        # Returns

        tuple: (head_model, trunk_model).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, X, head_model, trunk_model):
        """Get predictions from test data.

        This task corresponds to the creation of a composite testtuple on the Substra
        Platform.

        # Arguments

        X: testing data samples loaded with `Opener.get_X()`.
        head_model: head model loaded with `CompositeAlgo.load_head_model()`.
        trunk_model: trunk model loaded with `CompositeAlgo.load_trunk_model()`.

        # Returns

        predictions: predictions object.
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

    @abc.abstractmethod
    def load_head_model(self, path):
        """Deserialize head model from file.

        This method will be executed before the call to the methods
        `Algo.train()` and `Algo.predict()` to deserialize the model objects.

        # Arguments

        path: path of the model to load.

        # Returns

        model: the deserialized model object.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def save_head_model(self, model, path):
        """Serialize head model in file.

        This method will be executed after the call to the methods
        `Algo.train()` and `Algo.predict()` to save the model objects.

        # Arguments

        path: path of file to write.
        model: the model to serialize.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def load_trunk_model(self, path):
        """Deserialize trunk model from file.

        This method will be executed before the call to the methods
        `Algo.train()` and `Algo.predict()` to deserialize the model objects.

        # Arguments

        path: path of the model to load.

        # Returns

        model: the deserialized model object.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def save_trunk_model(self, model, path):
        """Serialize trunk model in file.

        This method will be executed after the call to the methods
        `Algo.train()` and `Algo.predict()` to save the model objects.

        # Arguments

        path: path of file to write.
        model: the model to serialize.
        """
        raise NotImplementedError


class CompositeAlgoWrapper(AlgoWrapper):
    """Algo wrapper to execute an algo instance on the platform."""
    _INTERFACE_CLASS = CompositeAlgo
    _DEFAULT_WORKSPACE_CLASS = CompositeAlgoWorkspace

    def _load_head_trunk_models(self, head_filename, trunk_filename):
        """Load head and trunk models from their filename."""
        head_model = None
        if head_filename:
            head_model_path = os.path.join(self._workspace.input_models_folder_path,
                                           head_filename)
            head_model = self._interface.load_head_model(head_model_path)
        trunk_model = None
        if trunk_filename:
            trunk_model_path = os.path.join(self._workspace.input_models_folder_path,
                                            trunk_filename)
            trunk_model = self._interface.load_trunk_model(trunk_model_path)
        return head_model, trunk_model

    def _assert_output_model_exists(self, path, part):
        if os.path.isdir(path):
            raise exceptions.NotAFileError(f'Expected output {part} model file at {path}, found dir')
        if not os.path.isfile(path):
            raise exceptions.MissingFileError(f'Output {part} model file {path} does not exists')

    def _assert_output_trunkmodel_exists(self):
        self._assert_output_model_exists(self._workspace.output_trunk_model_path, 'trunk')

    def _assert_output_headmodel_exists(self):
        self._assert_output_model_exists(self._workspace.output_head_model_path, 'head')

    def train(self, input_head_model_filename=None, input_trunk_model_filename=None,
              rank=0, fake_data=False, n_fake_samples=None):
        """Train method wrapper."""
        # load data from opener
        X = self._opener_wrapper.get_X(fake_data, n_fake_samples)
        y = self._opener_wrapper.get_y(fake_data, n_fake_samples)

        # load head and trunk models
        head_model, trunk_model = self._load_head_trunk_models(
            input_head_model_filename, input_trunk_model_filename)

        # train new models
        logger.info("launching training task")
        method = (self._interface.train if not fake_data else
                  self._interface._train_fake_data)
        head_model, trunk_model = method(X, y, head_model, trunk_model, rank)

        # serialize output head and trunk models and save them to workspace
        output_head_model_path = self._workspace.output_head_model_path
        logger.info("saving output head model to '{}'".format(output_head_model_path))
        self._interface.save_head_model(head_model, output_head_model_path)
        self._assert_output_headmodel_exists()

        output_trunk_model_path = self._workspace.output_trunk_model_path
        logger.info("saving output trunk model to '{}'".format(output_trunk_model_path))
        self._interface.save_trunk_model(trunk_model, output_trunk_model_path)
        self._assert_output_trunkmodel_exists()

        return head_model, trunk_model

    def predict(self, input_head_model_filename, input_trunk_model_filename,
                fake_data=False, n_fake_samples=None):
        """Predict method wrapper."""
        # load data from opener
        X = self._opener_wrapper.get_X(fake_data, n_fake_samples)

        # load head and trunk models
        head_model, trunk_model = self._load_head_trunk_models(
            input_head_model_filename, input_trunk_model_filename)
        assert head_model and trunk_model  # should not be None

        # get predictions
        logger.info("launching predict task")
        method = (self._interface.predict if not fake_data else
                  self._interface._predict_fake_data)
        pred = method(X, head_model, trunk_model)

        # save predictions
        self._opener_wrapper.save_predictions(pred)
        return pred


def _generate_composite_algo_cli(interface):
    """Helper to generate a command line interface client."""

    def _algo_from_args(args):
        workspace = CompositeAlgoWorkspace(
            input_data_folder_paths=args.data_sample_paths,
            input_models_folder_path=args.input_models_path,
            output_models_folder_path=args.output_models_path,
            output_head_model_filename=args.output_head_model_filename,
            output_trunk_model_filename=args.output_trunk_model_filename,
            log_path=args.log_path,
            output_predictions_path=args.output_predictions_path,
        )
        opener_wrapper = opener.load_from_module(
            path=args.opener_path,
            workspace=workspace,
        )
        utils.configure_logging(workspace.log_path, debug_mode=args.debug)
        return CompositeAlgoWrapper(
            interface,
            workspace=workspace,
            opener_wrapper=opener_wrapper,
        )

    def _parser_add_default_arguments(_parser):
        _parser.add_argument(
            '-d', '--fake-data', action='store_true', default=False,
            help="Enable fake data mode",
        )
        _parser.add_argument(
            '--n-fake-samples', default=None, type=int,
            help="Number of fake samples if fake data is used.",
        )
        _parser.add_argument(
            '--data-sample-paths', default=[],
            nargs='*',
            help="Define train/test data samples folder paths",
        )
        _parser.add_argument(
            '--input-models-path', default=None,
            help="Define input models folder path",
        )
        _parser.add_argument(
            '--output-predictions-path', default=None,
            help="Define output predictions file path",
        )
        _parser.add_argument(
            '--log-path', default=None,
            help="Define log filename path",
        )
        _parser.add_argument(
            '--opener-path', default=None,
            help="Define path to opener python script",
        )
        _parser.add_argument(
            '--debug', action='store_true', default=False,
            help="Enable debug mode (logs printed in stdout)",
        )
        # TODO the following options should be defined only for the train command
        _parser.add_argument(
            '--output-head-model-filename', type=str, default=None,
            help="Output head model filename (must be located in output models folder)"
        )
        _parser.add_argument(
            '--output-trunk-model-filename', type=str, default=None,
            help="Output trunk model filename (must be located in output models folder)"
        )
        _parser.add_argument(
            '--output-models-path', default=None,
            help="Define output models folder path",
        )

    def _train(args):
        algo_wrapper = _algo_from_args(args)
        algo_wrapper.train(
            args.input_head_model_filename,
            args.input_trunk_model_filename,
            args.rank,
            args.fake_data,
            args.n_fake_samples
        )

    parser = argparse.ArgumentParser()
    parsers = parser.add_subparsers()
    train_parser = parsers.add_parser('train')
    train_parser.add_argument(
        '--input-head-model-filename', type=str, default=None,
        help="Input head model filename (must be located in input models folder)"
    )
    train_parser.add_argument(
        '--input-trunk-model-filename', type=str, default=None,
        help="Input trunk model filename (must be located in input models folder)"
    )
    train_parser.add_argument(
        '-r', '--rank', type=int, default=0,
        help="Define machine learning task rank",
    )
    _parser_add_default_arguments(train_parser)
    train_parser.set_defaults(func=_train)

    def _predict(args):
        algo_wrapper = _algo_from_args(args)
        algo_wrapper.predict(
            args.input_head_model_filename,
            args.input_trunk_model_filename,
            args.fake_data,
            args.n_fake_samples
        )

    predict_parser = parsers.add_parser('predict')
    predict_parser.add_argument(
        '--input-head-model-filename', type=str, required=True,
        help="Input head model filename (must be located in input models folder)"
    )
    predict_parser.add_argument(
        '--input-trunk-model-filename', type=str, required=True,
        help="Input trunk model filename (must be located in input models folder)"
    )
    _parser_add_default_arguments(predict_parser)
    predict_parser.set_defaults(func=_predict)

    return parser


class AggregateAlgo(abc.ABC):
    """Abstract base class for defining an aggregate algo to run on the platform.

    To define a new aggregate algo script, subclass this class and implement the
    following abstract methods:

    - #AggregateAlgo.aggregate()
    - #AggregateAlgo.load_model()
    - #AggregateAlgo.save_model()

    This class has an `use_models_generator` class property:
    - if True, models will be passed to the `aggregate` method as a generator
    - (default) if False, models will be passed to the `aggregate` method as a list

    To add a aggregate algo to the Substra Platform, the line
    `tools.algo.execute(<AggregateAlgoClass>())` must be added to the main of the algo
    python script. It defines the aggregate algo command line interface and thus enables
    the Substra Platform to execute it.

    # Example

    ```python
    import json
    import substratools as tools


    class DummyAggregateAlgo(tools.AggregateAlgo):
        def aggregate(self, models, rank):
            new_model = None
            return new_model

        def load_model(self, path):
            return json.load(path)

        def save_model(self, model, path):
            json.dump(model, path)


    if __name__ == '__main__':
        tools.algo.execute(DummyAggregateAlgo())
    ```

    # How to test locally an aggregate algo script

    # Using the command line

    The aggregate algo script can be directly tested through it's command line interface.
    For instance to train an algo using fake data, run the following command:

    ```sh
    python <script_path> aggregate --models_path <models_path> --models <model_name> --model <model_name> --debug
    ```

    To see all the available options for the aggregate command, run:

    ```sh
    python <script_path> aggregate --help
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

    use_models_generator = False

    @abc.abstractmethod
    def aggregate(self, models, rank):
        """Aggregate models and produce a new model.

        This task corresponds to the creation of an aggregate tuple on the Substra
        Platform.

        # Arguments

        models: list of models loaded with `AggregateAlgo.load_model()`.
        rank: rank of the aggregate task.

        # Returns

        model: aggregated model.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def load_model(self, path):
        """Deserialize model from file.

        This method will be executed before the call to the method `Algo.aggregate()`
        to deserialize the model objects.

        # Arguments

        path: path of the model to load.

        # Returns

        model: the deserialized model object.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def save_model(self, model, path):
        """Serialize model in file.

        This method will be executed after the call to the method `Algo.aggregate()`
        to save the model objects.

        # Arguments

        path: path of file to write.
        model: the model to serialize.
        """
        raise NotImplementedError


class AggregateAlgoWrapper(object):
    """Aggregate algo wrapper to execute an aggregate algo instance on the platform."""
    _DEFAULT_WORKSPACE_CLASS = AggregateAlgoWorkspace

    def __init__(self, interface, workspace=None):
        assert isinstance(interface, AggregateAlgo)
        self._workspace = workspace or self._DEFAULT_WORKSPACE_CLASS()
        self._interface = interface

    def _assert_output_model_exists(self):
        path = self._workspace.output_model_path
        if os.path.isdir(path):
            raise exceptions.NotAFileError(f'Expected output model file at {path}, found dir')
        if not os.path.isfile(path):
            raise exceptions.MissingFileError(f'Output model file {path} does not exists')

    def _load_model(self, model_name):
        """Load single model in memory from its name."""
        # load model from workspace and deserialize it
        model_path = os.path.join(self._workspace.input_models_folder_path, model_name)
        logger.info("loading model from '{}'".format(model_path))
        return self._interface.load_model(model_path)

    def _load_models_as_list(self, model_names):
        return [self._load_model(model_name) for model_name in model_names]

    def _load_models_as_generator(self, model_names):
        for model_name in model_names:
            yield self._load_model(model_name)

    def _load_models(self, model_names):
        """Load models either as list or as generator"""
        if self._interface.use_models_generator:
            return self._load_models_as_generator(model_names)
        return self._load_models_as_list(model_names)

    def aggregate(self, model_names, rank=0):
        """Aggregate method wrapper."""
        # load models
        models = self._load_models(model_names)

        # train new model
        logger.info("launching aggregate task")
        model = self._interface.aggregate(models, rank)

        # serialize output model and save it to workspace
        logger.info("saving output model to '{}'".format(
            self._workspace.output_model_path))
        self._interface.save_model(model, self._workspace.output_model_path)
        self._assert_output_model_exists()
        return model


def _generate_aggregate_algo_cli(interface):
    """Helper to generate a command line interface client."""

    def _algo_from_args(args):
        workspace = AggregateAlgoWorkspace(
            input_models_folder_path=args.models_path,
            log_path=args.log_path,
            output_model_path=args.output_model_path,
        )
        utils.configure_logging(workspace.log_path, debug_mode=args.debug)
        return AggregateAlgoWrapper(
            interface,
            workspace=workspace,
        )

    def _parser_add_default_arguments(_parser):
        _parser.add_argument(
            '--models-path', default=None,
            help="Define models folder path",
        )
        _parser.add_argument(
            '--output-model-path', default=None,
            help="Define output model file path",
        )
        _parser.add_argument(
            '--log-path', default=None,
            help="Define log filename path",
        )
        _parser.add_argument(
            '--debug', action='store_true', default=False,
            help="Enable debug mode (logs printed in stdout)",
        )

    def _aggregate(args):
        algo_wrapper = _algo_from_args(args)
        algo_wrapper.aggregate(
            args.models,
            args.rank,
        )

    parser = argparse.ArgumentParser()
    parsers = parser.add_subparsers()
    aggregate_parser = parsers.add_parser('aggregate')
    aggregate_parser.add_argument(
        'models', type=str, nargs='*',
        help="Model names (must be located in default models folder)"
    )
    aggregate_parser.add_argument(
        '-r', '--rank', type=int, default=0,
        help="Define machine learning task rank",
    )
    _parser_add_default_arguments(aggregate_parser)
    aggregate_parser.set_defaults(func=_aggregate)

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
