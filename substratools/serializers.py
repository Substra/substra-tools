from collections import namedtuple
import json


Serializer = namedtuple('Serializer', ['loads', 'dumps'])

JSON = Serializer(json.loads, json.dumps)
