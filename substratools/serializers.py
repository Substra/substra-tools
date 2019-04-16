from collections import namedtuple
import json


Serializer = namedtuple('Serializer', ['loads', 'dumps'])

# TODO use load/dump by default
JSON = Serializer(json.loads, json.dumps)
