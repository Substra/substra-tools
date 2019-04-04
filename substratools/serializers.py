from collections import namedtuple
import json

Serializer = namedtuple('Serializer', ['load', 'dump'])

JSON = Serializer(json.loads, json.dumps)
