import abc
import json

from substratools import opener, workspace, utils


class Metrics(abc.ABC):
    OPENER = None

    @abc.abstractmethod
    def score(self, y_true, y_pred):
        """Returns the macro-average recall.

        Must return a float.
        """
        raise NotImplementedError


class MetricsWrapper(object):
    _OPENER_WRAPPER = None

    def __init__(self, interface):
        assert isinstance(interface, Metrics)

        if interface.OPENER:
            self._OPENER_WRAPPER = opener.OpenerWrapper(interface.OPENER)
        else:
            self._OPENER_WRAPPER = opener.load_from_module()
        assert isinstance(self._OPENER_WRAPPER, opener.OpenerWrapper)

        self._workspace = workspace.Workspace()
        self._interface = interface

    def _save_score(self, score):
        with open(self._workspace.score_filepath, 'w') as f:
            json.dump({'all': score}, f)

    def score(self):
        """Load labels and predictions and save score results."""
        y = self._OPENER_WRAPPER.get_y()
        y_pred = self._OPENER_WRAPPER.get_pred()
        x = self._interface.score(y, y_pred)
        self._save_score(x)
        return x


def _execute(interface):
    """Launch metrics script from interface."""
    wp = MetricsWrapper(interface)
    return wp.score()


def execute(module_name='metrics'):
    """Launch metrics script."""
    interface = utils.load_interface_from_module('metrics', Metrics)
    return _execute(interface)
