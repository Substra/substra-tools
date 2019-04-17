import imp
import importlib
import inspect
import logging
import sys


def import_module(module_name, code):
    if module_name in sys.modules:
        logging.warning(f'Module {module_name} will be overwritten')
    module = imp.new_module(module_name)
    sys.modules[module_name] = module
    exec(code, module.__dict__)


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
