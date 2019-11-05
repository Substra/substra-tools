# coding: utf8
import abc
import argparse
import logging
import os
import sys

from substratools import opener, utils
from substratools.workspace import Workspace


logger = logging.getLogger(__name__)


class Algo(abc.ABC):
    """Abstract base class for defining algo to run on the platform.

    To define a new algo script, subclass this class and implement the
    following abstract methods:

    - #Algo.train()
    - #Algo.predict()
    - #Algo.load_model()
    - #Algo.save_model()

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
            predictions = 0
            new_model = None
            return predictions, new_model

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

    The algo script can be directly tested through it's command line interface.
    For instance to train an algo using fake data, run the following command:

    ```sh
    python <script_path> train --fake-data --debug
    ```

    To see all the available options for the train and predict commands, run:

    ```sh
    python <script_path> train --help
    python <script_path> predict --help
    ```

    """

    @abc.abstractmethod
    def train(self, X, y, models, rank):
        """Train model and produce new model from train data.

        This task corresponds to the creation of a traintuple on the Substra
        Platform.

        # Arguments

        X: training data samples loaded with `Opener.get_X()`.
        y: training data samples labels loaded with `Opener.get_y()`.
        models: list of models loaded with `Algo.load_model()`.
        rank: rank of the training task.

        # Returns

        tuple: (predictions, model).
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
        method if you want to implement a different behaviour.
        """
        return self.train(*args, **kwargs)

    def _predict_fake_data(self, *args, **kwargs):
        """Predict model fake data mode.

        This method is called by the algorithm wrapper when the fake data mode
        is enabled. In fake data mode, `X` input arg has been replaced by
        the opener fake data.

        By default, it only calls directly `Algo.predict()` method. Override
        this method if you want to implement a different behaviour.
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

    def __init__(self, interface, workspace=None, opener_wrapper=None):
        assert isinstance(interface, Algo)
        self._workspace = workspace or Workspace()
        self._opener_wrapper = opener_wrapper or \
            opener.load_from_module(workspace=self._workspace)
        self._interface = interface

    def _load_models(self, model_names):
        """Load models in-memory from names."""
        # load models from workspace and deserialize them
        models = []
        models_path = self._workspace.input_models_folder_path
        logger.info("loading models from '{}'".format(models_path))
        for name in model_names:
            path = os.path.join(models_path, name)
            m = self._interface.load_model(path)
            models.append(m)
        return models

    def _save_model(self, model):
        """Save model object to workspace."""
        path = self._workspace.output_model_path
        logger.info("saving output model to '{}'".format(path))
        self._interface.save_model(model, path)

    def train(self, model_names, rank=0, fake_data=False):
        """Train method wrapper."""
        # load data from opener
        X = self._opener_wrapper.get_X(fake_data)
        y = self._opener_wrapper.get_y(fake_data)

        # load models
        models = self._load_models(model_names)

        # train new model
        logger.info("launching training task")
        method = (self._interface.train if not fake_data else
                  self._interface._train_fake_data)
        pred, model = method(X, y, models, rank)

        # serialize output model and save it to workspace
        self._save_model(model)

        # save predictions
        self._opener_wrapper.save_predictions(pred)

        return pred, model

    def predict(self, model_name, fake_data=False):
        """Predict method wrapper."""
        # load data from opener
        X = self._opener_wrapper.get_X(fake_data)

        # load models
        models = self._load_models([model_name])

        # get predictions
        logger.info("launching predict task")
        method = (self._interface.predict if not fake_data else
                  self._interface._predict_fake_data)
        pred = method(X, models[0])

        # save predictions
        self._opener_wrapper.save_predictions(pred)
        return pred


def _generate_cli(interface):
    """Helper to generate a command line interface client."""

    def _algo_from_args(args):
        workspace = Workspace(
            input_data_folder_path=args.data_samples_path,
            input_models_folder_path=args.models_path,
            log_path=args.log_path,
            output_model_path=args.output_model_path,
            output_predictions_path=args.output_predictions_path,
        )
        opener_wrapper = opener.load_from_module(
            path=args.opener_path,
            workspace=workspace,
        )
        utils.configure_logging(workspace.log_path, debug_mode=args.debug)
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
            '--data-samples-path', default=None,
            help="Define train/test data samples folder path",
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
        )

    predict_parser = parsers.add_parser('predict')
    predict_parser.add_argument(
        'model', type=str,
        help="Model name (must be located in default models folder)"
    )
    _parser_add_default_arguments(predict_parser)
    predict_parser.set_defaults(func=_predict)

    return parser


def execute(interface, sysargs=None):
    """Launch algo command line interface."""
    cli = _generate_cli(interface)
    sysargs = sysargs if sysargs is not None else sys.argv[1:]
    args = cli.parse_args(sysargs)
    args.func(args)
    return args
