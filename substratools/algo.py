import abc
import click


class Algo(abc.ABC):
    """Abstract base class for algo to submit on the platform."""

    @abc.abstractmethod
    def train(self, models, rank):
        """Train algorithm and produce new model."""
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, models):
        """Load a model and save predictions made on train data."""
        raise NotImplementedError

    @abc.abstractmethod
    def dry_run(self, models):
        """Train model dry run mode."""
        raise NotImplementedError


def _generate_cli(algo):
    """Helper to generate a command line interface client."""

    @click.group()
    @click.pass_context
    def cli(ctx):
        ctx.obj = algo

    @cli.command()
    @click.argument('models', nargs=-1)
    @click.pass_obj
    def dry_run(algo, models):
        algo.dry_run(models)

    @cli.command()
    @click.argument('models', nargs=-1)
    @click.option('-r', '--rank', type=click.INT, default=0,
                  help='Rank of the fltask')
    @click.pass_obj
    def train(algo, models, rank):
        algo.train(models, rank)

    @cli.command()
    @click.argument('models', nargs=-1)
    def predict(algo, models):
        algo.predict(models)

    return cli


def execute(algo):
    """Launch algo command line interface."""
    cli = _generate_cli(algo)
    cli()
