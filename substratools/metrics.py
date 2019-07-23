import abc
import argparse
import enum
import json
import logging
import sys

from substratools import opener, utils
from substratools.workspace import Workspace

logger = logging.getLogger(__name__)
REQUIRED_FUNCTIONS = set(['score'])


class DryRunMode(enum.IntEnum):
    DISABLED = 0
    FAKE_Y = 1
    FAKE_Y_PRED = 2

    @classmethod
    def from_value(cls, val):
        if isinstance(val, bool):
            # for backward compatibility with boolean dry_run values
            return cls.DISABLED if not val else cls.FAKE_Y_PRED
        return cls(val)

    @classmethod
    def from_str(cls, val):
        return DryRunMode[val]


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

    The metrics script can be directly tested through it's command line
    interface.  For instance to get the metrics from fake data, run the
    following command:

    ```sh
    python <script_path> --dry-run --debug
    ```

    To see all the available options for metrics commands, run:

    ```sh
    python <script_path> --help
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
        self._workspace = workspace or Workspace()
        self._opener_wrapper = opener_wrapper or \
            opener.load_from_module(workspace=self._workspace)
        self._interface = interface

    def _save_score(self, score):
        path = self._workspace.output_perf_path
        logger.info("saving score to '{}'".format(path))
        with open(path, 'w') as f:
            json.dump({'all': score}, f)

    def score(self, dry_run=False):
        """Load labels and predictions and save score results."""
        mode = DryRunMode.from_value(dry_run)
        if mode == DryRunMode.DISABLED:
            y = self._opener_wrapper.get_y()
            y_pred = self._opener_wrapper.get_predictions()

        elif mode == DryRunMode.FAKE_Y:
            y = self._opener_wrapper.get_y(dry_run=True)
            y_pred = self._opener_wrapper.get_predictions()

        elif mode == DryRunMode.FAKE_Y_PRED:
            y = self._opener_wrapper.get_y(dry_run=True)
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
        '-d', '--dry-run', action='store_true', default=False,
        help="Enable dry run mode (fake y)",
    )
    parser.add_argument(
        '--dry-run-mode', default=DryRunMode.DISABLED.name,
        choices=[e.name for e in DryRunMode],
        help="Set dry run mode",
    )
    parser.add_argument(
        '--data-samples-path', default=None,
        help="Define train/test data samples folder path",
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

    workspace = Workspace(
        input_data_folder_path=args.data_samples_path,
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
    dry_run = args.dry_run or DryRunMode.from_str(args.dry_run_mode)
    return metrics_wrapper.score(
        dry_run,
    )
