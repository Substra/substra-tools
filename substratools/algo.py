# coding: utf8
import abc
import argparse
from copy import deepcopy
import logging
import sys

from substratools import opener
from substratools import utils
from substratools.task_resources import TaskInputResources
from substratools.task_resources import TaskOutputResources
from substratools.workspace import GenericAlgoWorkspace


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


class GenericAlgo(abc.ABC):
    chainkeys_path = None


class GenericAlgoWrapper(object):
    """Generic wrapper to execute an algo instance on the platform."""

    _INTERFACE_CLASS = GenericAlgo

    def __init__(self, interface, workspace, opener_wrapper):
        assert isinstance(interface, self._INTERFACE_CLASS)
        self._workspace = workspace
        self._opener_wrapper = opener_wrapper
        self._interface = interface

        self._interface.chainkeys_path = self._workspace.chainkeys_path

    def _assert_outputs_exists(self, outputs: dict):
        pass

    @utils.Timer(logger)
    def task_launcher(self, method_name: str, rank: int = 0, fake_data: bool = False, n_fake_samples: int = None):
        """Launch Tasks"""

        # load inputs
        inputs = deepcopy(self._workspace.task_inputs)
        inputs.update({"rank": rank})

        # load data from opener
        if self._opener_wrapper is not None:
            X = self._opener_wrapper.get_X(fake_data, n_fake_samples)
            y = self._opener_wrapper.get_y(fake_data, n_fake_samples)

            if fake_data:
                logger.info("Using fake data with %i fake samples." % int(n_fake_samples))

            inputs.update({"X": X})
            inputs.update({"y": y})

        # load outputs
        outputs = deepcopy(self._workspace.task_outputs)

        # Retrieve method from user
        method = getattr(self._interface, method_name)

        logger.info("Launching task: executing `%s` function." % method_name)
        method(
            inputs=inputs,
            outputs=outputs,
        )

        self._assert_outputs_exists(
            self._workspace.task_outputs,
        )


def _generate_generic_algo_cli(interface):
    """Helper to generate a command line interface client."""

    def _algo_from_args(args):
        inputs = TaskInputResources(args.inputs)
        outputs = TaskOutputResources(args.outputs)
        log_path = args.log_path
        chainkeys_path = inputs.chainkeys_path
        compute_plan_path = inputs.compute_plan_path

        workspace = GenericAlgoWorkspace(
            log_path=log_path,
            chainkeys_path=chainkeys_path,
            compute_plan_path=compute_plan_path,
            inputs=inputs,
            outputs=outputs,
        )

        utils.configure_logging(workspace.log_path, log_level=args.log_level)

        opener_wrapper = opener.load_from_module(
            workspace=workspace,
        )

        return GenericAlgoWrapper(interface, workspace, opener_wrapper)

    def _user_func(args):
        algo_wrapper = _algo_from_args(args)
        algo_wrapper.task_launcher(
            method_name=args.method_name,
            rank=args.rank,
            fake_data=args.fake_data,
            n_fake_samples=args.n_fake_samples,
        )

    parser = argparse.ArgumentParser()
    _parser_add_default_arguments(parser)
    parser.set_defaults(func=_user_func)

    return parser


class Algo(GenericAlgo):
    @abc.abstractmethod
    def train(
        self,
        inputs: dict,
        outputs: dict,
    ) -> None:

        raise NotImplementedError

    @abc.abstractmethod
    def predict(
        self,
        inputs: dict,
        outputs: dict,
    ) -> None:

        raise NotImplementedError


class CompositeAlgo(GenericAlgo):
    @abc.abstractmethod
    def train(
        self,
        inputs: dict,
        outputs: dict,
    ) -> None:

        raise NotImplementedError

    @abc.abstractmethod
    def predict(
        self,
        inputs: dict,
        outputs: dict,
    ) -> None:

        raise NotImplementedError


class AggregateAlgo(GenericAlgo):
    @abc.abstractmethod
    def aggregate(
        self,
        inputs: dict,
        outputs: dict,
    ) -> None:

        raise NotImplementedError

    @abc.abstractmethod
    def predict(
        self,
        inputs: dict,
        outputs: dict,
    ) -> None:

        raise NotImplementedError


def execute(interface, sysargs=None):
    """Launch algo command line interface."""

    cli = _generate_generic_algo_cli(interface)

    sysargs = sysargs if sysargs is not None else sys.argv[1:]
    args = cli.parse_args(sysargs)
    args.func(args)
    return args
