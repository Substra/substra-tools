import abc
import logging
import os

import click

from substratools import serializers, opener, workspace


def _validate_serializer(serializer):
    assert isinstance(serializer, serializers.Serializer)
    assert callable(serializer.load)
    assert callable(serializer.dump)


class Algo(abc.ABC):
    """Abstract base class for defining algo to run on the platform."""
    OPENER = None
    MODEL_SERIALIZER = serializers.JSON  # default serializer

    @abc.abstractmethod
    def train(self, X, y, models, rank):
        """Train algorithm and produce new model.

        Must return a tuple (predictions, model).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, X, y, models):
        """Load models and save predictions made on train data.

        Must return predictions.
        """
        raise NotImplementedError

    def dry_run(self, X, y, models, rank):
        """Train model dry run mode."""
        return self.train(X, y, models, rank=0)


class AlgoWrapper(object):
    """Algo wrapper to execute an algo instance on the platform."""
    MODEL_SERIALIZER = None
    _OPENER_WRAPPER = None

    def __init__(self, interface):
        assert isinstance(interface, Algo)

        # validate or load default opener
        if interface.OPENER:
            self._OPENER_WRAPPER = opener.OpenerWrapper(interface.OPENER)
        else:
            self._OPENER_WRAPPER = opener.load_from_module()
        assert isinstance(self._OPENER_WRAPPER, opener.OpenerWrapper)

        # validate model serializer
        _validate_serializer(interface.MODEL_SERIALIZER)
        self.MODEL_SERIALIZER = interface.MODEL_SERIALIZER

        self._interface = interface
        self._workspace = workspace.Workspace()

    def _load_models(self, model_names):
        """Load models in-memory from names."""
        # load models from workspace and deserialize them
        models = []
        for name in model_names:
            path = os.path.join(self._workspace.model_folder, name)
            with open(path, 'r') as f:
                m = self.MODEL_SERIALIZER.load(f)
            models.append(m)
        return models

    def _save_model(self, model):
        """Save model object to workspace."""
        with open(self._workspace.output_model_filepath, 'w') as f:
            self.MODEL_SERIALIZER.dump(model, f)

    def train(self, model_names, rank=0, dry_run=False):
        """Train method wrapper."""
        # load data from opener
        logging.info('loading data from opener')
        X = self._OPENER_WRAPPER.get_X(dry_run)
        y = self._OPENER_WRAPPER.get_y(dry_run)

        # load models
        logging.info('loading models')
        models = self._load_models(model_names)

        # train new model
        logging.info('training')
        method = (self._interface.train if not dry_run else
                  self._interface.dry_run)
        pred, model = method(X, y, models, rank)

        # serialize output model and save it to workspace
        logging.info('saving output model')
        self._save_model(model)

        # save prediction
        logging.info('saving prediction')
        self._OPENER_WRAPPER.save_pred(pred)

        return pred, model

    def predict(self, model_names):
        """Predict method wrapper."""
        # load data from opener
        logging.info('loading data from opener')
        X = self._OPENER_WRAPPER.get_X()
        y = self._OPENER_WRAPPER.get_y()

        # load models
        logging.info('loading models')
        models = self._load_models(model_names)

        # get predictions
        logging.info('predicting')
        pred = self._interface.predict(X, y, models)

        # save prediction
        logging.info('saving prediction')
        self._OPENER_WRAPPER.save_pred(pred)

        return pred


def _generate_cli(interface):
    """Helper to generate a command line interface client."""
    algo_wrapper = AlgoWrapper(interface)
    logging.basicConfig(filename=algo_wrapper._workspace.log_path,
                        level=logging.DEBUG)

    @click.group()
    @click.pass_context
    def cli(ctx):
        ctx.obj = algo_wrapper

    @cli.command()
    @click.argument('models', nargs=-1)
    @click.option('-r', '--rank', type=click.INT, default=0,
                  help='Rank of the fltask')
    @click.option('-d', '--dry-run', is_flag=True,
                  help='Launch in dry run mode')
    @click.pass_obj
    def train(algo, models, rank, dry_run):
        algo.train(models, rank, dry_run)

    @cli.command()
    @click.argument('models', nargs=-1)
    @click.pass_obj
    def predict(algo, models):
        algo.predict(models)

    return cli


def execute(interface):
    """Launch algo command line interface."""
    cli = _generate_cli(interface)
    cli()
