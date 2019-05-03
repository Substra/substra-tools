import abc
import enum
import json

from substratools import opener, workspace, utils


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


class Metrics(abc.ABC):
    @abc.abstractmethod
    def score(self, y_true, y_pred):
        """Returns the macro-average recall.

        Must return a float.
        """
        raise NotImplementedError


class MetricsWrapper(object):

    def __init__(self, interface):
        self._opener_wrapper = opener.load_from_module()

        self._workspace = workspace.Workspace()
        self._interface = interface

    def _save_score(self, score):
        with open(self._workspace.score_filepath, 'w') as f:
            json.dump({'all': score}, f)

    def score(self, dry_run=False):
        """Load labels and predictions and save score results."""
        mode = DryRunMode.from_value(dry_run)
        if mode == DryRunMode.DISABLED:
            y = self._opener_wrapper.get_y()
            y_pred = self._opener_wrapper.get_pred()

        elif mode == DryRunMode.FAKE_Y:
            y = self._opener_wrapper.get_y(dry_run=True)
            y_pred = self._opener_wrapper.get_pred()

        elif mode == DryRunMode.FAKE_Y_PRED:
            y = self._opener_wrapper.get_y(dry_run=True)
            y_pred = y

        else:
            raise AssertionError

        x = self._interface.score(y, y_pred)
        self._save_score(x)
        return x


def _execute(interface, dry_run):
    """Launch metrics script from interface."""
    wp = MetricsWrapper(interface)
    return wp.score(dry_run=dry_run)


def execute(module_name='metrics', dry_run=False):
    """Launch metrics script."""
    interface = utils.load_interface_from_module(
        module_name,
        interface_class=Metrics,
        interface_signature=REQUIRED_FUNCTIONS)
    return _execute(interface, dry_run)
