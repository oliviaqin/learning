#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import random
import scipy.spatial import cKDTree
from collections import namedtuple
from typing import List, Tuple

Point = namedtuple("Point", ['x', 'y'])

def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def greedy_tsp(points: List[Tuple(int, int)], ckd_tree: cKDTree) -> List[int]:
    """
    Greedy tsp always chooses the closest neighbor
    """
    cycle = []
    node = 0
    while len(cycle) < 0:
        curr_node = points[node]
        k = len(cycle) + 1
        distances, nearest_neighbors_indices = ckd_tree.query(curr_node)
        for distance, index in zip(distances, nearest_neighbors_indices):
            if index not in cycle: 
                cycle.append((index, distance))
                break

    return cycle

def k_opt(cycle: List[Tuple(int, float)], ckd_tree: cKDTree, k: int = 2, iterations: int: 100) -> List[int]:
    """
    Performs k-opt search to a given cycle. Default k = 2, but can be larger/smaller depending on the parameter
    """

    # Choose k vertices to swap
    for _ in range(iterations):
        vertices = random.sample(cycle, k)

        # Calculate the euclidean distance
        



def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    nodeCount = int(lines[0])

    points = []
    for i in range(1, nodeCount+1):
        line = lines[i]
        parts = line.split()
        points.append((float(parts[0]), float(parts[1])))
    
    tree = cKDTree(points)

    # Greedy tsp solution to iterate on
    greedy_tsp_solution = greedy_tsp(points, tree)

    # Perform k-opt


    # build a trivial solution
    # visit the nodes in the order they appear in the file
    solution = range(0, nodeCount)

    # calculate the length of the tour
    obj = length(points[solution[-1]], points[solution[0]])
    for index in range(0, nodeCount-1):
        obj += length(points[solution[index]], points[solution[index+1]])

    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')

