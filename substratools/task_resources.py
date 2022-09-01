import json
from typing import Dict
from typing import List
from typing import Optional

from substratools import exceptions

# TODO: share those constant with backend
TASK_IO_PREDICTIONS = "predictions"
TASK_IO_OPENER = "opener"
TASK_IO_CHAINKEYS = "chainkeys"
TASK_IO_DATASAMPLES = "datasamples"
TRAIN_IO_MODELS = "models"
TRAIN_IO_MODEL = "model"
COMPOSITE_IO_SHARED = "shared"
COMPOSITE_IO_LOCAL = "local"

_RESOURCE_ID = "id"
_RESOURCE_VALUE = "value"


class TaskResources:
    """TaskResources is created from stdin to provide a nice abstraction over inputs/outputs"""

    # TODO: we may want to pass List[str] instead
    _values: Dict[str, List[str]]

    def __init__(self, argstr: str) -> None:
        """Create outputs from argument string

        Argument is expected to be a JSON array like:
        [
            {"id": "local", "value": "/sandbox/output/model/uuid"},
            {"id": "shared", ...}
        ]
        """
        self._values = {}

        argument_list = json.loads(argstr)

        for item in argument_list:
            self._values.setdefault(item[_RESOURCE_ID], list()).append(item[_RESOURCE_VALUE])

    def get_value(self, key: str) -> str:
        """Return a value corresponding to the given key.
        It will raise if there is no input for given key or if there are multiple values"""
        val = self._values[key]
        if len(val) > 1:
            raise exceptions.InvalidInputOutputsError("there are more than one path")

        return val[0]

    def get_values(self, key: str) -> List[str]:
        """get_values will return the list of str corresponding to the given key.
        It will raise if key does not exist.
        """
        return self._values[key]

    def get_optional_value(self, key: str) -> Optional[str]:
        """Return value for given key, won't raise if there is no matching resource.
        Will raise if there are more than one value."""
        if key not in self._values:
            return None
        val = self._values[key]

        if len(val) > 1:
            raise exceptions.InvalidInputOutputsError("there are more than one path")

        return val[0]

    def get_optional_values(self, key: str) -> Optional[List[str]]:
        """Return values for given key, won't raise if there is no matching resource."""
        return self._values.get(key)
