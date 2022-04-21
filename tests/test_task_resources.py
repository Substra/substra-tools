import json

from substratools.task_resources import TaskResources


def test_multiple_resources():
    resources = [
        {"id": "test", "value": "1"},
        {"id": "test", "value": "2"},
        {"id": "single", "value": "42"},
    ]

    resources = TaskResources(json.dumps(resources))

    assert resources.get_values("test") == ["1", "2"]
    assert resources.get_value("single") == "42"
    assert resources.get_optional_values("test") == ["1", "2"]
    assert resources.get_optional_value("single") == "42"
    assert resources.get_optional_values("non-existant") is None
    assert resources.get_optional_value("non-existant") is None
