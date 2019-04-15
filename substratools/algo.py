import abc
import logging

import click

from substratools import serializers, opener, workspace


def _validate_serializer(serializer):
    assert isinstance(serializer, serializers.Serializer)
    assert callable(serializer.loads)
    assert callable(serializer.dumps)


class Algo(abc.ABC):
    """Abstract base class for executing an algo on the platform."""
    OPENER = None
    MODEL_SERIALIZER = serializers.JSON  # default serializer

    _OPENER_WRAPPER = None

    def __init__(self):
        if self.OPENER:
            self._OPENER_WRAPPER = opener.OpenerWrapper(self.OPENER)
        else:
            self._OPENER_WRAPPER = opener.load_from_module()
        assert isinstance(self._OPENER_WRAPPER, opener.OpenerWrapper)

        self._workspace = workspace.Workspace()

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
        X = self._OPENER_WRAPPER.get_X(dry_run)
        y = self._OPENER_WRAPPER.get_y(dry_run)

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
        self._OPENER_WRAPPER.save_pred(pred)

        return pred, model

    def _execute_predict(self, model_paths):
        """Predict method wrapper."""
        # load data from opener
        logging.info('loading data from opener')
        X = self._OPENER_WRAPPER.get_X()
        y = self._OPENER_WRAPPER.get_y()

        # load models
        logging.info('loading models')
        models = self._load_models(model_paths)

        # get predictions
        logging.info('predicting')
        pred = self.predict(X, y, models)

        # save prediction
        logging.info('saving prediction')
        self._OPENER_WRAPPER.save_pred(pred)

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
