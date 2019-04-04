import abc
import os

import importlib
import click

from substratools import serializers


def _load_opener_module():
    """Load opener module based on current working directory."""
    return importlib.import_module('opener')


class Workspace(object):
    """Filesystem workspace for algo execution."""

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

    def save_model(self, buff, name='model'):
        with open(os.path.join(self._model_folder, name), 'w') as f:
            return f.write(buff)

    def load_model(self, name='model'):
        with open(os.path.join(self._model_folder, name), 'r') as f:
            return f.read()


def _validate_serializer(serializer):
    assert isinstance(serializer, serializers.Serializer)
    assert callable(serializer.load)
    assert callable(serializer.dump)


class Algo(abc.ABC):
    """Abstract base class for algo to submit on the platform."""
    OPENER = None
    MODEL_SERIALIZER = serializers.JSON

    def __init__(self):
        if self.OPENER is None:
            self.OPENER = _load_opener_module()
        self._workspace = Workspace()

        _validate_serializer(self.MODEL_SERIALIZER)

        super(Algo, self).__init__()

    @abc.abstractmethod
    def train(self, X, y, models, rank):
        """Train algorithm and produce new model."""
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, models):
        """Load a model and save predictions made on train data."""
        raise NotImplementedError

    def dry_run(self, X, y, models, rank):
        """Train model dry run mode."""
        return self.train(X, y, models, rank=0)

    def _execute_train(self, model_paths, rank=0, dry_run=False):
        """Train method wrapper."""
        # load data from opener
        if dry_run:
            X = self.OPENER.fake_X()
            y = self.OPENER.fake_y()
        else:
            X = self.OPENER.get_X()
            y = self.OPENER.get_y()

        # load models from workspace and deserialize them
        model_buffers = [self._workspace.load_model(path)
                         for path in model_paths]
        models = [self.MODEL_SERIALIZER.load(buff) for buff in model_buffers]

        # train new model
        method = self.train if not dry_run else self.dry_run
        pred, model = method(X, y, models, rank)

        # serialize output model and save it to workspace
        model_buff = self.MODEL_SERIALIZER.dump(model)
        self._workspace.save_model(model_buff)

        # save prediction
        self.OPENER.save_pred(pred)

        return pred, model


def _generate_cli(algo):
    """Helper to generate a command line interface client."""

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
    def predict(algo, models):
        algo.predict(models)

    return cli


def execute(algo):
    """Launch algo command line interface."""
    cli = _generate_cli(algo)
    cli()
