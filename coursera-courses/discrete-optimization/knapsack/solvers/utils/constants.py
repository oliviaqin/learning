from collections import namedtuple

Item = namedtuple("Item", ['index', 'value', 'weight'])
Node = namedtuple("Node", ['value', 'taken', 'capacity', 'heuristic_value'])
