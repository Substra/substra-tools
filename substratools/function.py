import inspect
from functools import wraps

from substratools import algo


def tools_function(method):
    @wraps(method)
    def wrap_and_call(_skip: bool = False, sysargs=None):
        if "inputs" not in inspect.signature(method).parameters:
            raise

        if "outputs" not in inspect.signature(method).parameters:
            raise

        if "task_properties" not in inspect.signature(method).parameters:
            raise

        if _skip:
            return method
        else:

            return algo.execute(method, sysargs=sysargs)

    return wrap_and_call
