import imp
import importlib
import inspect
import logging
import os
import sys

from substratools import exceptions


def configure_logging(path=None, debug_mode=False):
    kwargs = {}
    level = logging.DEBUG
    if path and not debug_mode:
        kwargs['filename'] = path

    logging.basicConfig(level=level, **kwargs)

    if debug_mode:
        # set root logger level in case the root logger has already handlers
        # configured for it
        logging.getLogger('substratools').setLevel(level)


def import_module(module_name, code):
    if module_name in sys.modules:
        logging.warning("Module {} will be overwritten".format(module_name))
    module = imp.new_module(module_name)
    sys.modules[module_name] = module
    exec(code, module.__dict__)


def import_module_from_path(path, module_name):
    assert os.path.exists(path), "path '{}' not found".format(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec, "could not load spec from path '{}'".format(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_interface_from_module(module_name, interface_class,
                               interface_signature=None, path=None):
    if path:
        module = import_module_from_path(path, module_name)
    else:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            # XXX don't use ModuleNotFoundError for python3.5 compatibility
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
