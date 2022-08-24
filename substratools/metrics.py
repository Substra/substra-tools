import abc
import argparse
import json
import logging
import os
import sys
from typing import Any
from typing import TypedDict

from substratools import exceptions
from substratools import opener
from substratools import utils
from substratools.algo import InputIdentifiers
from substratools.algo import OutputIdentifiers
from substratools.workspace import MetricsWorkspace

logger = logging.getLogger(__name__)
REQUIRED_FUNCTIONS = set(["score"])


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
        def score(self, inputs, outputs):
            y_true = inputs["y"]
            y_pred = self.load_predictions(inputs["predictions"])
            perf = accuracy_score(y_true, y_pred)
            tools.save_performance(perf, outputs["performance"])

        def load_predictions(self, predictions_path):
            return json.load(predictions_path)

    if __name__ == '__main__':
         tools.metrics.execute(AccuracyMetrics())
    ```

    # How to test locally a metrics script

    # Using the command line

    The metrics script can be directly tested through it's command line
    interface.  For instance to get the metrics from fake data, run the
    following command:

    ```sh
    python <script_path> --fake-data --n-fake-samples 20 --log-level debug
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

    inputs = {"y": y_true, "predictions":predictions_path}
    outputs = {"performance": performance_path}
    m.score(inputs, outputs)
    ```
    """

    @abc.abstractmethod
    def score(
        self,
        inputs: TypedDict("inputs", {InputIdentifiers.y: Any, InputIdentifiers.predictions: os.PathLike}),
        outputs: TypedDict("outputs", {OutputIdentifiers.performance: os.PathLike}),
    ):
        """Compute model perf from actual and predicted values.

        # Arguments

        inputs: TypedDict(
            "inputs",
            {
                InputIdentifiers.y: Any: actual values.
                InputIdentifiers.predictions: Any: path to predicted values.
            }
        ),
        outputs: TypedDict(
            "outputs",
            {
                OutputIdentifiers.performance: os.PathLike: path to save the performance of the model.
            }
        )
        """
        raise NotImplementedError


class MetricsWrapper(object):
    def __init__(self, interface, workspace=None, opener_wrapper=None):
        self._workspace = workspace or MetricsWorkspace()
        self._opener_wrapper = opener_wrapper or opener.load_from_module(workspace=self._workspace)
        self._interface = interface

    def _save_score(self, score):
        path = self._workspace.output_perf_path
        logger.info("saving score to '{}'".format(path))
        with open(path, "w") as f:
            json.dump({"all": score}, f)

    def _assert_output_exists(self, path, key):

        if os.path.isdir(path):
            raise exceptions.NotAFileError(f"Expected output file at {path}, found dir for output `{key}`")
        if not os.path.isfile(path):
            raise exceptions.MissingFileError(f"Output file {path} used to save argument `{key}` does not exists.")

    def score(self, fake_data=False, n_fake_samples=None):
        """Load labels and predictions and save score results."""
        y_pred_path = self._workspace.input_predictions_path

        if not fake_data:
            y = self._opener_wrapper.get_y()

        elif fake_data:
            y = self._opener_wrapper.get_y(fake_data=True, n_fake_samples=n_fake_samples)

        logger.info("launching scoring task")

        inputs = {InputIdentifiers.y: y, InputIdentifiers.predictions: y_pred_path}
        outputs = {OutputIdentifiers.performance: self._workspace.output_perf_path}

        self._interface.score(inputs, outputs)

        self._assert_output_exists(self._workspace.output_perf_path, OutputIdentifiers.performance)


def _generate_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--fake-data",
        action="store_true",
        default=False,
        help="Enable fake data mode (fake y)",
    )
    parser.add_argument(
        "--n-fake-samples",
        type=int,
        default=None,
        help="Number of fake samples if fake data is used.",
    )
    parser.add_argument(
        "--data-sample-paths",
        default=[],
        nargs="*",
        help="Define train/test data samples folder paths",
    )
    parser.add_argument(
        "--input-predictions-path",
        default=None,
        help="Define input predictions file path",
    )
    parser.add_argument(
        "--output-perf-path",
        default=None,
        help="Define output perf file path",
    )
    parser.add_argument(
        "--opener-path",
        default=None,
        help="Define path to opener python script",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=utils.MAPPING_LOG_LEVEL.keys(),
        help="Choose log level",
    )
    parser.add_argument(
        "--log-path",
        default="pred/metrics.log",
        help="Define log filename path",
    )

    return parser


def save_performance(performance: Any, path: os.PathLike):
    with open(path, "w") as f:
        json.dump({"all": performance}, f)


def load_performance(path: os.PathLike) -> Any:
    with open(path, "r") as f:
        performance = json.load(f)["all"]
    return performance


def execute(interface=None, sysargs=None):
    """Launch metrics command line interface."""
    if not interface:
        interface = utils.load_interface_from_module(
            "metrics", interface_class=Metrics, interface_signature=REQUIRED_FUNCTIONS
        )

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
    utils.configure_logging(path=workspace.log_path, log_level=args.log_level)
    metrics_wrapper = MetricsWrapper(
        interface,
        workspace=workspace,
        opener_wrapper=opener_wrapper,
    )
    fake_data = args.fake_data
    n_fake_samples = args.n_fake_samples
    metrics_wrapper.score(
        fake_data,
        n_fake_samples,
    )

    return metrics_wrapper._workspace.output_perf_path
