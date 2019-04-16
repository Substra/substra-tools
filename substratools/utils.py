import importlib
import inspect
import sys


def load_interface_from_module(module_name, interface_class):
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        raise

    # check interface
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and issubclass(obj, interface_class):
            return obj()  # return interface instance

    # backward compatibility; accept module
    return module
