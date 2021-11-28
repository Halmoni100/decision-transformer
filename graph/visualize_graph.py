import os, sys
sys.path.append(os.environ['DT_ROOT'])

import networkx as nx
import matplotlib.pyplot as plt
import graph_util

import constants

graph = graph_util.load_current_graph()

target_node = constants.GOAL
shortest_path_lengths = []
for n in graph.nodes:
    if n != target_node:
        shortest_path_lengths.append(graph.shortest_path_length(n, target_node))
plt.hist(shortest_path_lengths)
plt.show()

G = nx.Graph()
for n in graph.nodes:
    for n_neighbor in graph.get_neighbors(n):
        G.add_edge(n, n_neighbor)
nx.draw(G, with_labels=True)
plt.show()
