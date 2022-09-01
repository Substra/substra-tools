import json
from typing import Dict
from typing import List
from typing import Optional
from substratools import exceptions

TASK_IO_OPENER = "opener"
TASK_IO_DATASAMPLES = "datasamples"

TASK_IO_PREDICTIONS = "predictions"
TASK_IO_LOCALFOLDER = "localfolder"
TASK_IO_CHAINKEYS = "chainkeys"
TRAIN_IO_MODELS = "models"
TRAIN_IO_MODEL = "model"
COMPOSITE_IO_SHARED = "shared"
COMPOSITE_IO_LOCAL = "local"

_RESOURCE_ID = "id"
_RESOURCE_VALUE = "value"
_RESOURCE_MULTIPLE = "multiple"


class TaskResources:
    """TaskResources is created from stdin to provide a nice abstraction over inputs/outputs"""

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
            self._values.setdefault(
                item[_RESOURCE_ID], dict(_RESOURCE_VALUE=[], _RESOURCE_MULTIPLE=item[_RESOURCE_MULTIPLE])
            )
            self._values[item[_RESOURCE_ID]][_RESOURCE_VALUE].append(item[_RESOURCE_ID])

    def get_optional_value(self, key: str) -> Optional[str]:
        """Return value for given key, won't raise if there is no matching resource.
        Will raise if there are more than one value."""
        if key not in self._values:
            return None
        val = self._values[key][_RESOURCE_VALUE]

        if len(val) > 1:
            raise exceptions.InvalidInputOutputsError("there are more than one path")

        return val[0]

    def get_optional_values(self, key: str) -> Optional[List[str]]:
        """Return values for given key, won't raise if there is no matching resource."""
        return self._values.get(key, {}).get(_RESOURCE_VALUE)


class TaskInputResources(TaskResources):
    def __init__(self, argstr: str) -> None:
        super().__init__(argstr)

        self.opener_path = self.get_optional_value(TASK_IO_OPENER)
        self.input_data_folder_paths = self.get_optional_values(TASK_IO_DATASAMPLES)
        self.chainkeys_path = self.get_optional_value(TASK_IO_CHAINKEYS)

        self.task_inputs = {
            k: v
            for k, v in self._values.items()
            if k not in (TASK_IO_OPENER, TASK_IO_DATASAMPLES, TASK_IO_LOCALFOLDER, TASK_IO_CHAINKEYS)
        }


class TaskOutputResources(TaskResources):
    def __init__(self, argstr: str) -> None:
        super().__init__(argstr)
        # If there is only one path, the user expects a Path not a list
        self.task_outputs = self._values
