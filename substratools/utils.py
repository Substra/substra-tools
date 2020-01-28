import imp
import importlib
import inspect
import logging
import os
import sys

from substratools import exceptions


logger = logging.getLogger(__name__)


def configure_logging(path=None, debug_mode=True):
    level = logging.DEBUG if debug_mode else logging.INFO

    formatter = logging.Formatter('%(name)s - %(message)s')

    h = logging.StreamHandler()
    h.setLevel(level)
    h.setFormatter(formatter)

    root = logging.getLogger('substratools')
    root.setLevel(level)
    root.addHandler(h)

    if path and debug_mode:
        fh = logging.FileHandler(path)
        fh.setLevel(level)
        fh.setFormatter(formatter)

        root.addHandler(h)


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
        logger.info(f"Module '{module_name}' loaded from path '{path}'")
    else:
        try:
            module = importlib.import_module(module_name)
            logger.info(
                f"Module '{module_name}' imported dynamically; module={module}")
        except ImportError:
            # XXX don't use ModuleNotFoundError for python3.5 compatibility
            raise

    # check if module empty
    if not inspect.getmembers(
            module, lambda m: inspect.isclass(m) or inspect.isfunction(m)):
        raise exceptions.EmptyInterface(
            f"Module '{module_name}' seems empty: no method/class found in members: '{dir(module)}'")

    # find interface class
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, interface_class):
            return obj()  # return interface instance

    # backward compatibility; accept methods at module level directly
    if interface_signature is None:
        class_name = interface_class.__name__
        elements = str(dir(module))
        logger.info(f"Class '{class_name}' not found from: '{elements}'")
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
