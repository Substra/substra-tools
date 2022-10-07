# coding: utf8
import argparse
import json
import logging
import os
import sys
from copy import deepcopy
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional

from substratools import exceptions
from substratools import opener
from substratools import utils
from substratools.exceptions import FunctionNotFoundError
from substratools.task_resources import StaticInputIdentifiers
from substratools.task_resources import TaskResources
from substratools.workspace import FunctionWorkspace

logger = logging.getLogger(__name__)


def _parser_add_default_arguments(parser):
    parser.add_argument(
        "--function-name",
        type=str,
        help="The name of the function to execute from the given file",
    )
    parser.add_argument(
        "-r",
        "--rank",
        type=int,
        default=0,
        help="Define machine learning task rank",
    ),
    parser.add_argument(
        "-d",
        "--fake-data",
        action="store_true",
        default=False,
        help="Enable fake data mode",
    )
    parser.add_argument(
        "--n-fake-samples",
        default=None,
        type=int,
        help="Number of fake samples if fake data is used.",
    )
    parser.add_argument(
        "--log-path",
        default=None,
        help="Define log filename path",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=utils.MAPPING_LOG_LEVEL.keys(),
        help="Choose log level",
    )
    parser.add_argument(
        "--inputs",
        type=str,
        default="[]",
        help="Inputs of the compute task",
    )
    parser.add_argument(
        "--outputs",
        type=str,
        default="[]",
        help="Outputs of the compute task",
    )


class FunctionWrapper(object):
    """Wrapper to execute a function on the platform."""

    def __init__(self, workspace: FunctionWorkspace, opener_wrapper: Optional[opener.OpenerWrapper]):
        self._workspace = workspace
        self._opener_wrapper = opener_wrapper

    def _assert_outputs_exists(self, outputs: Dict[str, str]):
        for key, path in outputs.items():
            if os.path.isdir(path):
                raise exceptions.NotAFileError(f"Expected output file at {path}, found dir for output `{key}`")
            if not os.path.isfile(path):
                raise exceptions.MissingFileError(f"Output file {path} used to save argument `{key}` does not exists.")

    @utils.Timer(logger)
    def execute(self, function: Callable, rank: int = 0, fake_data: bool = False, n_fake_samples: int = None):
        """Execute a compute task"""

        # load inputs
        inputs = deepcopy(self._workspace.task_inputs)

        task_properties = {StaticInputIdentifiers.rank.value: rank}

        # load data from opener
        if self._opener_wrapper:
            loaded_datasamples = self._opener_wrapper.get_data(fake_data, n_fake_samples)

            if fake_data:
                logger.info("Using fake data with %i fake samples." % int(n_fake_samples))

            assert (
                StaticInputIdentifiers.datasamples.value not in inputs.keys()
            ), f"{StaticInputIdentifiers.datasamples.value} must be an input of kind `datasamples`"
            inputs.update({StaticInputIdentifiers.datasamples.value: loaded_datasamples})

        # load outputs
        outputs = deepcopy(self._workspace.task_outputs)

        logger.info("Launching task: executing `%s` function." % function.__name__)
        function(
            inputs=inputs,
            outputs=outputs,
            task_properties=task_properties,
        )

        self._assert_outputs_exists(
            self._workspace.task_outputs,
        )


def _generate_function_cli():
    """Helper to generate a command line interface client."""

    def _function_from_args(args):
        inputs = TaskResources(args.inputs)
        outputs = TaskResources(args.outputs)
        log_path = args.log_path
        chainkeys_path = inputs.chainkeys_path

        workspace = FunctionWorkspace(
            log_path=log_path,
            chainkeys_path=chainkeys_path,
            inputs=inputs,
            outputs=outputs,
        )

        utils.configure_logging(workspace.log_path, log_level=args.log_level)

        opener_wrapper = opener.load_from_module(
            workspace=workspace,
        )

        return FunctionWrapper(workspace, opener_wrapper)

    def _user_func(args, function):
        function_wrapper = _function_from_args(args)
        function_wrapper.execute(
            function=function,
            rank=args.rank,
            fake_data=args.fake_data,
            n_fake_samples=args.n_fake_samples,
        )

    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
    _parser_add_default_arguments(parser)
    parser.set_defaults(func=_user_func)

    return parser


def _get_function_from_name(functions, function_name):

    for function in functions:
        if function.__name__ == function_name:
            return function

    raise FunctionNotFoundError(f"The function {function_name} given as --function-name argument as not been found.")


def save_performance(performance: Any, path: os.PathLike):
    with open(path, "w") as f:
        json.dump({"all": performance}, f)


def load_performance(path: os.PathLike) -> Any:
    with open(path, "r") as f:
        performance = json.load(f)["all"]
    return performance


def execute(*functions, sysargs=None):
    """Launch function command line interface."""

    cli = _generate_function_cli()

    sysargs = sysargs if sysargs is not None else sys.argv[1:]
    args = cli.parse_args(sysargs)
    function = _get_function_from_name(functions, args.function_name)
    args.func(args, function)
    return args
