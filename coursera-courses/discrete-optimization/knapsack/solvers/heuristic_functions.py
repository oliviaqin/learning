from typing import List, Callable
from collections import namedtuple

Item = namedtuple("Item", ['index', 'value', 'weight'])

class HeuristicFunctions:
    """
    Class with heuristic functions to bound the search tree.
    """

    @staticmethod
    def optimistic_total(node, depth: int, items:List[Item]) -> int:
        """
        Attitude is everything
        :return:
        """
        return node.value + sum([item.value for item in items[depth:]])

    @staticmethod
    def belgian_chocolate(node, depth:int, items: List[Item]) -> int:
        """
        We can split things like belgian chocolate.
        :return:
        """
        density_sorted_items = sorted(items[depth:], key=lambda item: item.value/item.weight, reverse=True)
        value = node.value
        capacity = node.capacity
        while capacity > 0 and len(density_sorted_items) > 0:
            item = density_sorted_items.pop(0)
            if item.weight > capacity:
                value += capacity/item.weight*item.value
                capacity = 0
            else:
                value += item.value
                capacity -= item.weight
        return value