import numpy as np
from typing import List, Tuple, Callable
# from utils.constants import Item, Node
from collections import namedtuple, deque
import time

Item = namedtuple("Item", ['index', 'value', 'weight'])
DFSNode = namedtuple("DFSNode", ['value', 'taken', 'capacity', 'depth'])
BFSNode = namedtuple("BFSNode", ['value', 'taken', 'capacity', 'depth'])

TIME_BOUND = 15

def remove_ith_item(items: List[Item], index: int) -> List[Item]:
    return [item for item in items if item.index != index]

def dfs(items: List[Item], capacity: int, heuristic_function: Callable, threshold: float) -> Tuple[int, List[int], int]:
    """

    :param items:
    :param capacity:
    :param heuristic_function:
    :param threshold
    :param prune
    :return:
    """
    # the depth of the tree correspond to the i'th item and if we include it in the knapsack
    # left = we include the item
    # right = we do not include the item
    # Always go left first until we cannot

    nodes_traversed = 0
    optimal = 1
    root_node = DFSNode(0, [], capacity, 0)
    best_node = DFSNode(-1, [], capacity, 0)

    start_time = time.time()

    # Sort items by the density
    items.sort(key=lambda item: item.value/item.weight, reverse=True)

    queue = deque()
    queue.append(root_node)

    while len(queue) > 0:
        node = queue.pop()

        if time.time() - start_time > TIME_BOUND:
            optimal = 0
            break

        if node.capacity < 0:
            # Invalid nodes have value -1 for heuristic + value
            continue

        if node.depth == len(items):
            # Leaf node
            best_node = max(best_node, node, key=lambda x: x.value)

        if len(items) > 400:
            optimal = 0
            best_node = max(best_node, node, key=lambda x: x.value)

        if node.depth < len(items) and heuristic_function(node, node.depth, items) > best_node.value:
            current_item = items[node.depth]

            # We include the item
            queue.append(DFSNode(node.value + current_item.value, node.taken + [current_item.index], node.capacity - current_item.weight, node.depth + 1))

            # We don't include the item
            queue.append(DFSNode(node.value, node.taken, node.capacity, node.depth + 1))

    taken = [0]*len(items)
    for node_index in best_node.taken:
        taken[node_index] = 1

    return best_node.value, taken, optimal

def bfs(items: List[Item], capacity: int, heuristic_function: Callable) -> Tuple[int, List[int], int]:

    root_node = best_node = BFSNode(0, [], capacity, 0)
    queue = deque([root_node])

    items.sort(key=lambda item: item.value/item.weight, reverse=True)

    nodes_traversed = 0
    optimal = 1

    start_time = time.time()

    while len(queue) > 0:
        node = queue.popleft()

        nodes_traversed += 1

        if time.time() - start_time > TIME_BOUND:
            optimal = 0
            break

        if node.capacity < 0:
            continue

        if node.depth == len(items):
            best_node = max(best_node, node, key=lambda x: x.value)
            continue

        if len(items) > 400:
            optimal = 0
            best_node = max(best_node, node, key=lambda x: x.value)

        if node.depth < len(items) and heuristic_function(node, node.depth, items) > best_node.value:
            current_item = items[node.depth]

            # We include the item
            queue.append(BFSNode(node.value + current_item.value, node.taken + [current_item.index], node.capacity - current_item.weight, node.depth + 1))

            # We don't include the item
            queue.append(BFSNode(node.value, node.taken, node.capacity, node.depth + 1))

    # print(f"BFS Nodes Traversed: {nodes_traversed}")

    taken = [0]*len(items)

    for index in best_node.taken:
        taken[index] = 1

    return best_node.value, taken, optimal




def branch_and_bound(items: List[Item], capacity: int, heuristic_function: Callable) -> Tuple[int, List[int]]:
    """
    Non-optimal solution based on a heuristic function and search method.
    Search methods implemented:
    - dfs
    - bfs
    -

    :param items:
    :param capacity:
    :param heuristic_function:
    :return:
    """



def trivial_greedy_knapsack(items: List[Item], capacity: int) -> Tuple[int, List[int]]:
    """
    Trivial greedy solution solver for the knapsack problem. Originally given in the assignment.
    :param items:
    :param capacity:
    :return:
    """
    value = 0
    weight = 0
    taken = [0]*len(items)

    for item in items:
        if weight + item.weight <= capacity:
            taken[item.index] = 1
            value += item.value
            weight += item.weight

    return (value, taken)

def dynamic_programming_solution(items: List[Item], capacity: int) -> Tuple[int, List[int]]:
    """
    DP implementation to solve the knapsack problem. Good for small # of items. Finds the optimal solution

    :param items: List of Item tuples
    :param capacity: The capacity of the knapsack
    :return: Tuple(value, taken)
    """
    item_count = len(items)
    dp_table = np.zeros((capacity+1, item_count), dtype=int)  # initialize a item_count x capacity table

    for i in range(item_count):
        for j in range(capacity+1):
            # Best solution for the current grid
            if items[i].weight <= j and i > 0:
                dp_table[j][i] = max(dp_table[j][i-1], items[i].value + dp_table[j - items[i].weight][i-1])
            elif i == 0 and items[i].weight <= j:
                dp_table[j][i] = items[i].value
            else:
                dp_table[j][i] = dp_table[j][i-1]

    best_value = dp_table[capacity][item_count-1]

    pointer_1 = capacity
    pointer_2 = item_count-1

    taken = np.zeros(len(items), dtype=int)

    # Calculate taken
    while pointer_2 >=0 and pointer_1 >= 0:
        if pointer_2 == 0:
            taken[pointer_2] = dp_table[pointer_1][pointer_2] > 0
        elif dp_table[pointer_1][pointer_2-1] == dp_table[pointer_1][pointer_2]:
            taken[pointer_2] = 0
        else:
            taken[pointer_2] = 1
            pointer_1 -= items[pointer_2].weight
        pointer_2 -= 1

    return (best_value, taken)