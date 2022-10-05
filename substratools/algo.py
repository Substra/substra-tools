# coding: utf8
import abc
import argparse
import json
import logging
import os
import sys
from copy import deepcopy
from typing import Any, Callable
from typing import Dict
from typing import Optional

from substratools import exceptions
from substratools import opener
from substratools import utils
from substratools.task_resources import StaticInputIdentifiers
from substratools.task_resources import TaskResources
from substratools.workspace import AlgoWorkspace

logger = logging.getLogger(__name__)


def _parser_add_default_arguments(parser):
    parser.add_argument(
        "--method-name",
        type=str,
        help="The name of the method to execute from the given file",
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


class GenericAlgoWrapper(object):
    """Generic wrapper to execute an algo instance on the platform."""

    def __init__(self, workspace: AlgoWorkspace, opener_wrapper: Optional[opener.OpenerWrapper]):
        self._workspace = workspace
        self._opener_wrapper = opener_wrapper

    def _assert_outputs_exists(self, outputs: Dict[str, str]):
        for key, path in outputs.items():
            if os.path.isdir(path):
                raise exceptions.NotAFileError(f"Expected output file at {path}, found dir for output `{key}`")
            if not os.path.isfile(path):
                raise exceptions.MissingFileError(f"Output file {path} used to save argument `{key}` does not exists.")

    @utils.Timer(logger)
    def execute(self, method: Callable, rank: int = 0, fake_data: bool = False, n_fake_samples: int = None):
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

        # Retrieve method from user
        # method = self._get_method_from_name(self._methods_list, method_name)

        logger.info("Launching task: executing `%s` function." % method.__name__)
        method(
            inputs=inputs,
            outputs=outputs,
            task_properties=task_properties,
        )

        self._assert_outputs_exists(
            self._workspace.task_outputs,
        )


def _generate_generic_algo_cli():
    """Helper to generate a command line interface client."""

    def _algo_from_args(args):
        inputs = TaskResources(args.inputs)
        outputs = TaskResources(args.outputs)
        log_path = args.log_path
        chainkeys_path = inputs.chainkeys_path

        workspace = AlgoWorkspace(
            log_path=log_path,
            chainkeys_path=chainkeys_path,
            inputs=inputs,
            outputs=outputs,
        )

        utils.configure_logging(workspace.log_path, log_level=args.log_level)

        opener_wrapper = opener.load_from_module(
            workspace=workspace,
        )

        return GenericAlgoWrapper(workspace, opener_wrapper)

    def _user_func(args, method):
        algo_wrapper = _algo_from_args(args)
        algo_wrapper.execute(
            method=method,
            rank=args.rank,
            fake_data=args.fake_data,
            n_fake_samples=args.n_fake_samples,
        )

    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
    _parser_add_default_arguments(parser)
    parser.set_defaults(func=_user_func)

    return parser


def _get_method_from_name(methods_list, method_name):
    for method in methods_list:
        if method.__name__ == method_name:
            return method
    raise


def save_performance(performance: Any, path: os.PathLike):
    with open(path, "w") as f:
        json.dump({"all": performance}, f)


def load_performance(path: os.PathLike) -> Any:
    with open(path, "r") as f:
        performance = json.load(f)["all"]
    return performance


def execute(methods_list, sysargs=None):
    """Launch algo command line interface."""

    cli = _generate_generic_algo_cli()

    sysargs = sysargs if sysargs is not None else sys.argv[1:]
    args = cli.parse_args(sysargs)
    method = _get_method_from_name(methods_list, args.method_name)
    args.func(args, method)
    return args
