import imp
import importlib
import inspect
import logging
import sys

from substratools import exceptions


def import_module(module_name, code):
    if module_name in sys.modules:
        logging.warning("Module {} will be overwritten".format(module_name))
    module = imp.new_module(module_name)
    sys.modules[module_name] = module
    exec(code, module.__dict__)


def load_interface_from_module(module_name, interface_class,
                               interface_signature=None):
    try:
        module = importlib.import_module(module_name)
    except ImportError:  # XXX don't use ModuleNotFoundError for python3.5 compatibility
        raise

    # check interface
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and issubclass(obj, interface_class):
            return obj()  # return interface instance

    # backward compatibility; accep
    if interface_signature is None:
        class_name = interface_class.__name__
        raise exceptions.InvalidInterface(
            "Expecting {} subclass in {}".format(
                class_name, module_name))

    missing_functions = interface_signature.copy()
    for name, obj in inspect.getmembers(module):
        if not inspect.isfunction(obj):
            continue
        try:
            missing_functions.remove(name)
        except KeyError:
            pass

    if missing_functions:
        message = "Method(s) {} not implemented".format(
            ", ".join(["'{}'".format(m) for m in missing_functions]))
        raise exceptions.InvalidInterface(message)
    return module
