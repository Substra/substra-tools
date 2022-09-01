import json
from typing import Dict
from typing import List
from typing import Optional
from typing import Union
from substratools import exceptions

TASK_IO_OPENER = "opener"
TASK_IO_DATASAMPLES = "datasamples"
TASK_IO_CHAINKEYS = "chainkeys"


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
            {"id": "local", "value": "/sandbox/output/model/uuid", "multiple": False},
            {"id": "shared", ...}
        ]
        """
        self._values = {}

        argument_list = json.loads(argstr)

        for item in argument_list:
            self._values.setdefault(
                item[_RESOURCE_ID], {_RESOURCE_VALUE: [], _RESOURCE_MULTIPLE: item[_RESOURCE_MULTIPLE]}
            )
            self._values[item[_RESOURCE_ID]][_RESOURCE_VALUE].append(item[_RESOURCE_VALUE])

        self.opener_path = self.get_value(TASK_IO_OPENER)
        self.input_data_folder_paths = self.get_value(TASK_IO_DATASAMPLES)
        self.chainkeys_path = self.get_value(TASK_IO_CHAINKEYS)

    def get_value(self, key: str) -> Optional[Union[List[str], str]]:
        """Returns the value for a given key. Won't raise if there is no matching resource.
        Will raise if there is a mismatch between the given multiplicity and teh number of returned
        elements.

        If multiple is True, will return a list else will return a single value
        """
        if key not in self._values:
            return None

        val = self._values[key][_RESOURCE_VALUE]
        multiple = self._values[key][_RESOURCE_MULTIPLE]

        if not multiple and len(val) > 1:
            raise exceptions.InvalidInputOutputsError(
                f"There is more than one path for the non multiple resource {key}"
            )

        if multiple:
            return val

        return val[0]

    @property
    def formatted_dynamic_resources(self) -> Union[List[str], str]:
        """Returns all the resources (except the datasamples, the opener and the chainkeys_path under the user format:
        A dict where each input is an element where
            - the key is the user identifier
            - the value is a list of Path for multiple resources and a Path for non multiple resources
        """

        return {
            k: self.get_value(k)
            for k in self._values.keys()
            if k not in (TASK_IO_OPENER, TASK_IO_DATASAMPLES, TASK_IO_CHAINKEYS)
        }
