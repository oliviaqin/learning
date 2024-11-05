#!/usr/bin/python
# -*- coding: utf-8 -*-
import time

from typing import List, Dict
from collections import deque


def propagate_vertex_swap(colors: List, color_chosen: int, color_to_swap_to: int, graph: Dict[int, List[int]]) -> List[int]:
    """
    Returns new colors array with highest_color_vertex swapped with it's adjacent nodes
    :param colors:
    :param highest_color_vertex:
    :return:
    """

    vertices_to_process = deque()

    # We swap the color of the highest color vertex to the color_to_swap_to
    impacted_vertices = [i for i in range(len(colors)) if colors[i] == color_chosen]

    # Preprocess vertexes
    for vertex in impacted_vertices:
        if any(colors[neighbor] == color_to_swap_to for neighbor in graph[vertex]):
            # Conflict present, plan for propagation
            vertices_to_process.append(vertex)
        else:
            # Safe to swap
            colors[vertex] = color_to_swap_to

    while len(vertices_to_process) > 0:
        vertex = vertices_to_process.popleft()

        # If there's an available color less than the current color
        available_colors = set(range(color_chosen)) - {colors[neighbor] for neighbor in graph[vertex]}

        if available_colors:
            color = min(available_colors)
            colors[vertex] = color

            for neighbor in graph[vertex]:
                if colors[neighbor] == color:
                    vertices_to_process.append(neighbor)

    return colors




def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])

    edges = []
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append((int(parts[0]), int(parts[1])))

    # build a trivial solution
    # every node has its own color
    # solution = range(0, node_count)

    # Psuedo code
    # order the edges by the vertex (i.e. order by [0]) with vertex with most edges first (appears the most in the list)
    # That way we can do more pruning (i.e. if a vertex with many outgoing edges is colored first, then we can get rid of more options for surrounding vertices)
    # Enforce lexographic ordering of solution so that we break symmetry
    # Domain = {0...|V|}
    # Decision variables = {0....|V-1|} -> representing the color of the nth node
    # Brute force: Color the vertex -> remove the value from the rest of the possible edges
    # Iteratively assign colors to each vertex

    colors = [-1]*node_count
    graph = {}

    # Create the graph.
    # We add all neighbors
    for edge in edges:
        node = edge[0]
        neighbor = edge[1]
        graph[node] = graph.get(node, []) + [neighbor]
        graph[neighbor] = graph.get(neighbor, []) + [node]

    # Sort the graph based on the number of neighbors
    sorted_keys_by_edges = sorted(graph, key=lambda k: len(graph[k]), reverse=True)


    # Iterate through the vertices using the greedy approach
    for vertex in sorted_keys_by_edges:
        # if a colors[vertex] = -1, then we have already assigned it a color and we skip
        if colors[vertex] == -1:
            possible_colors = set(range(node_count)) - {colors[neighbor] for neighbor in graph[vertex] if colors[neighbor] != -1 }
            # Assign the least possible value in domain for each vertex
            colors[vertex] = min(possible_colors)

        for neighbor in graph[vertex]:
            if len(graph[neighbor]) == 1:
                colors[neighbor] = 1

    # Check if the answer is correct
    for vertex in sorted_keys_by_edges:
        assigned_color = colors[vertex]
        for neighbor in graph[vertex]:
            if colors[neighbor] == assigned_color:
                print(f"bad solution")
                break

    start_time = time.time()

    print(max(colors))
    min_so_far = max(colors)

    # try:
    #     # Iteratively improve the answer by selecting a vertex and changing it another color
    #     for color_1 in range(min_so_far)[::-1]:
    #         for color_2 in range(min_so_far):
    #             # Skip similar colors
    #             if color_1 == color_2:
    #                 continue
    #             if time.time() - start_time > 60:
    #                 raise TimeoutError()
    #             colors = min(propagate_vertex_swap(colors, color_1, color_2, graph), colors, key=lambda x: max(x))
    # except TimeoutError as e:
    #     print(e)

    print(max(colors))


    # prepare the solution in the specified output format
    output_data = str(node_count) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, colors))

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
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')

