# coding: utf8
import abc
import argparse
import logging
import os
import sys

from substratools import opener, workspace


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

        def predict(self, X, y, model):
            predictions = 0
            return predictions

        def load_model(self, path):
            return json.load(path)

        def save_model(self, model, path):
            json.dump(model, path)


    if __name__ == '__main__':
        tools.algo.execute(DummyAlgo())
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

    def _train_dry_run(self, *args, **kwargs):
        """Train model dry run mode.

        This methods is called by the algorithm wrapper when the dry run mode
        is enabled. In dry run mode, `X` and `y` input args have been replaced
        by the opener fake data.

        By default, it only calls directly `Algo.train()` method. Override this
        method if you want to implement a different behaviour.
        """
        return self.train(*args, **kwargs)

    def _predict_dry_run(self, *args, **kwargs):
        """Predict model dry run mode.

        This methods is called by the algorithm wrapper when the dry run mode
        is enabled. In dry run mode, `X` input arg has been replaced by
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

    def __init__(self, interface):
        assert isinstance(interface, Algo)
        self._opener_wrapper = opener.load_from_module()

        self._interface = interface
        self._workspace = workspace.Workspace()

    def _load_models(self, model_names):
        """Load models in-memory from names."""
        # load models from workspace and deserialize them
        models = []
        for name in model_names:
            path = os.path.join(self._workspace.model_folder, name)
            m = self._interface.load_model(path)
            models.append(m)
        return models

    def _save_model(self, model):
        """Save model object to workspace."""
        self._interface.save_model(model,
                                   self._workspace.output_model_filepath)

    def train(self, model_names, rank=0, dry_run=False):
        """Train method wrapper."""
        # load data from opener
        logging.info('loading data from opener')
        X = self._opener_wrapper.get_X(dry_run)
        y = self._opener_wrapper.get_y(dry_run)

        # load models
        logging.info('loading models')
        models = self._load_models(model_names)

        # train new model
        logging.info('training')
        method = (self._interface.train if not dry_run else
                  self._interface._train_dry_run)
        pred, model = method(X, y, models, rank)

        # serialize output model and save it to workspace
        logging.info('saving output model')
        self._save_model(model)

        # save predictions
        logging.info('saving predictions')
        self._opener_wrapper.save_predictions(pred)

        return pred, model

    def predict(self, model_name, dry_run=False):
        """Predict method wrapper."""
        # load data from opener
        logging.info('loading data from opener')
        X = self._opener_wrapper.get_X(dry_run)

        # load models
        logging.info('loading models')
        models = self._load_models([model_name])

        # get predictions
        logging.info('predicting')
        method = (self._interface.predict if not dry_run else
                  self._interface._predict_dry_run)
        pred = method(X, models[0])

        # save predictions
        logging.info('saving predictions')
        self._opener_wrapper.save_predictions(pred)

        return pred


def _generate_cli(algo_wrapper):
    """Helper to generate a command line interface client."""

    def _train(args):
        algo_wrapper.train(args.models, args.rank, args.dry_run)

    def _predict(args):
        algo_wrapper.predict(args.model, args.dry_run)

    parser = argparse.ArgumentParser()
    parsers = parser.add_subparsers()

    train_parser = parsers.add_parser('train')
    train_parser.add_argument(
        'models', type=str, nargs='*')
    train_parser.add_argument(
        '-d', '--dry-run', action='store_true', default=False)
    train_parser.add_argument(
        '-r', '--rank', type=int, default=0)
    train_parser.set_defaults(func=_train)

    predict_parser = parsers.add_parser('predict')
    predict_parser.add_argument(
        'model', type=str)
    predict_parser.add_argument(
        '-d', '--dry-run', action='store_true', default=False)
    predict_parser.set_defaults(func=_predict)
    return parser


def execute(interface, sysargs=None):
    """Launch algo command line interface."""
    algo_wrapper = AlgoWrapper(interface)
    logging.basicConfig(filename=algo_wrapper._workspace.log_path,
                        level=logging.DEBUG)

    cli = _generate_cli(algo_wrapper)

    sysargs = sysargs if sysargs is not None else sys.argv[1:]
    logging.debug('launching command with: {}'.format(sysargs))
    args = cli.parse_args(sysargs)
    args.func(args)
    return args
