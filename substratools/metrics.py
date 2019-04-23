import abc
import json

from substratools import opener, workspace, utils


class Metrics(abc.ABC):
    @abc.abstractmethod
    def score(self, y_true, y_pred):
        """Returns the macro-average recall.

        Must return a float.
        """
        raise NotImplementedError


class MetricsWrapper(object):

    def __init__(self, interface):
        assert isinstance(interface, Metrics)

        self._opener_wrapper = opener.load_from_module()

        self._workspace = workspace.Workspace()
        self._interface = interface

    def _save_score(self, score):
        with open(self._workspace.score_filepath, 'w') as f:
            json.dump({'all': score}, f)

    def score(self, dry_run=False):
        """Load labels and predictions and save score results."""
        y = self._opener_wrapper.get_y(dry_run=dry_run)
        y_pred = y if dry_run else self._opener_wrapper.get_pred()
        x = self._interface.score(y, y_pred)
        self._save_score(x)
        return x


def _execute(interface, dry_run):
    """Launch metrics script from interface."""
    wp = MetricsWrapper(interface)
    return wp.score(dry_run=dry_run)


def execute(module_name='metrics', dry_run=False):
    """Launch metrics script."""
    interface = utils.load_interface_from_module('metrics', Metrics)
    return _execute(interface, dry_run)
