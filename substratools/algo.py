import abc
import logging
import os

import importlib
import click

from substratools import serializers


def _load_opener_module():
    """Load opener module based on current working directory."""
    return importlib.import_module('opener')


class Workspace(object):
    """Filesystem workspace for algo execution."""
    LOG_FILENAME = 'log_model.log'

    def __init__(self, dirpath=None):
        self._root_path = dirpath if dirpath else os.getcwd()

        self._data_folder = os.path.join(self._root_path, 'data')
        self._pred_folder = os.path.join(self._root_path, 'pred')
        self._model_folder = os.path.join(self._root_path, 'model')

        paths = [self._data_folder, self._pred_folder, self._model_folder]
        for path in paths:
            try:
                os.makedirs(path)
            except FileExistsError:
                pass

    @property
    def log_path(self):
        return os.path.join(self._model_folder, self.LOG_FILENAME)

    def save_model(self, buff, name='model'):
        with open(os.path.join(self._model_folder, name), 'w') as f:
            return f.write(buff)

    def load_model(self, name='model'):
        with open(os.path.join(self._model_folder, name), 'r') as f:
            return f.read()


def _validate_serializer(serializer):
    assert isinstance(serializer, serializers.Serializer)
    assert callable(serializer.loads)
    assert callable(serializer.dumps)


class Algo(abc.ABC):
    """Abstract base class for executing an algo on the platform."""
    OPENER = None
    MODEL_SERIALIZER = serializers.JSON  # default serializer

    def __init__(self):
        if self.OPENER is None:
            self.OPENER = _load_opener_module()
        self._workspace = Workspace()

        _validate_serializer(self.MODEL_SERIALIZER)

        super(Algo, self).__init__()

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

    def _load_models(self, model_paths):
        """Load models in-memory from paths."""
        # load models from workspace and deserialize them
        model_paths = model_paths if model_paths else []
        model_buffers = [self._workspace.load_model(path)
                         for path in model_paths]
        return [self.MODEL_SERIALIZER.loads(buff) for buff in model_buffers]

    def _save_model(self, model):
        """Save model object to workspace."""
        model_buff = self.MODEL_SERIALIZER.dumps(model)
        self._workspace.save_model(model_buff)

    def _execute_train(self, model_paths, rank=0, dry_run=False):
        """Train method wrapper."""
        # load data from opener
        logging.info('loading data from opener')
        if dry_run:
            X = self.OPENER.fake_X()
            y = self.OPENER.fake_y()
        else:
            X = self.OPENER.get_X()
            y = self.OPENER.get_y()

        # load models
        logging.info('loading models')
        models = self._load_models(model_paths)

        # train new model
        logging.info('training')
        method = self.train if not dry_run else self.dry_run
        pred, model = method(X, y, models, rank)

        # serialize output model and save it to workspace
        logging.info('saving output model')
        self._save_model(model)

        # save prediction
        logging.info('saving prediction')
        self.OPENER.save_pred(pred)

        return pred, model

    def _execute_predict(self, model_paths):
        """Predict method wrapper."""
        # load data from opener
        logging.info('loading data from opener')
        X = self.OPENER.get_X()
        y = self.OPENER.get_y()

        # load models
        logging.info('loading models')
        models = self._load_models(model_paths)

        # get predictions
        logging.info('predicting')
        pred = self.predict(X, y, models)

        # save prediction
        logging.info('saving prediction')
        self.OPENER.save_pred(pred)

        return pred


def _generate_cli(algo):
    """Helper to generate a command line interface client."""
    logging.basicConfig(filename=algo._workspace.log_path,
                        level=logging.DEBUG)

    @click.group()
    @click.pass_context
    def cli(ctx):
        ctx.obj = algo

    @cli.command()
    @click.argument('models', nargs=-1)
    @click.option('-r', '--rank', type=click.INT, default=0,
                  help='Rank of the fltask')
    @click.option('-d', '--dry-run', is_flag=True,
                  help='Launch in dry run mode')
    @click.pass_obj
    def train(algo, models, rank, dry_run):
        algo._execute_train(models, rank, dry_run)

    @cli.command()
    @click.argument('models', nargs=-1)
    @click.pass_obj
    def predict(algo, models):
        algo._execute_predict(models)

    return cli


def execute(algo):
    """Launch algo command line interface."""
    cli = _generate_cli(algo)
    cli()
