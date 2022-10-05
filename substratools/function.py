import inspect
import json
from functools import wraps

from substratools import algo


def function(method):
    @wraps(method)
    def wrap_and_call(_skip: bool = False, inputs=None, outputs=None, task_properties=None):
        if "inputs" not in inspect.signature(method).parameters:
            raise

        if "outputs" not in inspect.signature(method).parameters:
            raise

        if "task_properties" not in inspect.signature(method).parameters:
            raise

        if _skip:
            method(inputs=inputs, outputs=outputs)
        else:
            command = None

            if inputs is not None and outputs is not None:
                options = [
                    "--inputs",
                    json.dumps(inputs),
                    "--outputs",
                    json.dumps(outputs),
                ]

                command = ["--method-name", method.__name__]
                command.extend(options)

            return algo.execute(method, sysargs=command)

    return wrap_and_call
