from substratools import exceptions, Metrics
from substratools.utils import import_module, load_interface_from_module

import pytest


def test_invalid_interface():
    code = """
def score():
    pass
"""
    import_module('score', code)
    with pytest.raises(exceptions.InvalidInterface):
        load_interface_from_module('score', interface_class=Metrics)
