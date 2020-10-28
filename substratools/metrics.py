import abc
import argparse
import enum
import json
import logging
import sys

from substratools import opener, utils
from substratools.workspace import MetricsWorkspace

logger = logging.getLogger(__name__)
REQUIRED_FUNCTIONS = set(['score'])


class FakeDataMode(enum.IntEnum):
    DISABLED = 0
    FAKE_Y = 1
    FAKE_Y_PRED = 2

    @classmethod
    def from_value(cls, val):
        if isinstance(val, bool):
            # for backward compatibility with boolean fake_data values
            return cls.DISABLED if not val else cls.FAKE_Y_PRED
        return cls(val)

    @classmethod
    def from_str(cls, val):
        return FakeDataMode[val]


class Metrics(abc.ABC):
    """Abstract base class for defining the objective metrics.

    To define a new metrics, subclass this class and implement the
    unique following abstract method #Metrics.score().

    To add an objective to the Substra Platform, the line
    `tools.algo.execute(<MetricsClass>())` must be added to the main of the
    metrics python script. It defines the metrics command line interface and
    thus enables the Substra Platform to execute it.

    # Example

    ```python
    from sklearn.metrics import accuracy_score
    import substratools as tools


    class AccuracyMetrics(tools.Metrics):
        def score(self, y_true, y_pred):
            return accuracy_score(y_true, y_pred)

    if __name__ == '__main__':
         tools.metrics.execute(AccuracyMetrics())
    ```

    # How to test locally a metrics script

    # Using the command line

    The metrics script can be directly tested through it's command line
    interface.  For instance to get the metrics from fake data, run the
    following command:

    ```sh
    python <script_path> --fake-data --n-fake-samples 20 --debug
    ```

    To see all the available options for metrics commands, run:

    ```sh
    python <script_path> --help
    ```

    # Using a python script

    A metrics class can be imported and used in python scripts as would any other class.

    For example, assuming that you have files named `opener.py` and `metrics.py` that contains
    an `Opener` named  `MyOpener` and a `Metrics` called `MyMetrics`:

    ```python
    import os
    import opener
    import metrics

    o = MyOpener()
    m = MyMetrics()


    data_sample_folders = os.listdir('./sandbox/data_samples/')
    predictions_path = './sandbox/predictions'

    y_true = o.get_y(data_sample_folders)
    y_pred = o.get_predictions(predictions_path)

    score = m.score(y_true, y_pred)
    ```
    """

    @abc.abstractmethod
    def score(self, y_true, y_pred):
        """Compute model perf from actual and predicted values.

        # Arguments

        y_true: actual values.
        y_pred: predicted values.

        # Returns

        perf (float): performance of the model.
        """
        raise NotImplementedError


class MetricsWrapper(object):

    def __init__(self, interface, workspace=None, opener_wrapper=None):
        self._workspace = workspace or MetricsWorkspace()
        self._opener_wrapper = opener_wrapper or \
            opener.load_from_module(workspace=self._workspace)
        self._interface = interface

    def _save_score(self, score):
        path = self._workspace.output_perf_path
        logger.info("saving score to '{}'".format(path))
        with open(path, 'w') as f:
            json.dump({'all': score}, f)

    def score(self, fake_data=False, n_fake_samples=None):
        """Load labels and predictions and save score results."""
        mode = FakeDataMode.from_value(fake_data)
        if mode == FakeDataMode.DISABLED:
            y = self._opener_wrapper.get_y()
            y_pred = self._opener_wrapper.get_predictions()

        elif mode == FakeDataMode.FAKE_Y:
            y = self._opener_wrapper.get_y(fake_data=True, n_fake_samples=n_fake_samples)
            y_pred = self._opener_wrapper.get_predictions()

        elif mode == FakeDataMode.FAKE_Y_PRED:
            y = self._opener_wrapper.get_y(fake_data=True, n_fake_samples=n_fake_samples)
            y_pred = y

        else:
            raise AssertionError

        logger.info("launching scoring task")
        x = self._interface.score(y, y_pred)
        logger.info("score: {}".format(x))
        self._save_score(x)
        return x


def _generate_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--fake-data', action='store_true', default=False,
        help="Enable fake data mode (fake y)",
    )
    parser.add_argument(
        '--fake-data-mode', default=FakeDataMode.DISABLED.name,
        choices=[e.name for e in FakeDataMode],
        help="Set fake data mode",
    )
    parser.add_argument(
        '--n-fake-samples', type=int, default=None,
        help="Number of fake samples if fake data is used.",
    )
    parser.add_argument(
        '--data-sample-paths', default=[],
        nargs='*',
        help="Define train/test data samples folder paths",
    )
    parser.add_argument(
        '--input-predictions-path', default=None,
        help="Define input predictions file path",
    )
    parser.add_argument(
        '--output-perf-path', default=None,
        help="Define output perf file path",
    )
    parser.add_argument(
        '--opener-path', default=None,
        help="Define path to opener python script",
    )
    parser.add_argument(
        '--debug', action='store_true', default=False,
        help="Enable debug mode (logs printed in stdout)",
    )
    parser.add_argument(
        '--log-path', default='pred/metrics.log',
        help="Define log filename path",
    )
    return parser


def execute(interface=None, sysargs=None):
    """Launch metrics command line interface."""
    if not interface:
        interface = utils.load_interface_from_module(
            'metrics',
            interface_class=Metrics,
            interface_signature=REQUIRED_FUNCTIONS)

    cli = _generate_cli()
    sysargs = sysargs if sysargs is not None else sys.argv[1:]
    args = cli.parse_args(sysargs)

    workspace = MetricsWorkspace(
        input_data_folder_paths=args.data_sample_paths,
        input_predictions_path=args.input_predictions_path,
        log_path=args.log_path,
        output_perf_path=args.output_perf_path,
    )
    opener_wrapper = opener.load_from_module(
        path=args.opener_path,
        workspace=workspace,
    )
    utils.configure_logging(path=workspace.log_path, debug_mode=args.debug)
    metrics_wrapper = MetricsWrapper(
        interface,
        workspace=workspace,
        opener_wrapper=opener_wrapper,
    )
    fake_data = args.fake_data or FakeDataMode.from_str(args.fake_data_mode)
    n_fake_samples = args.n_fake_samples
    return metrics_wrapper.score(
        fake_data,
        n_fake_samples,
    )
