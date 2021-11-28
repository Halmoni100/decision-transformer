import os
import random
import pickle
from collections import deque

class Graph:
    def __init__(self, nodes):
        self.edges = {}
        self.nodes = nodes
        for n in nodes:
            self.edges[n] = set()

    def get_nodes(self):
        return self.nodes

    def add_edge(self, edge):
        a, b = edge
        self.edges[a].add(b)
        self.edges[b].add(a)

    def add_random_edges(self, num_edges):
        edges_to_add = num_edges
        while edges_to_add > 0:
            rand_idx_a = random.randint(0, len(self.nodes) - 1)
            node_a = self.nodes[rand_idx_a]
            rand_idx_b = random.randint(0, len(self.nodes) - 1)
            node_b = self.nodes[rand_idx_b]
            if node_a != node_b and node_b not in self.edges[node_a]:
                self.edges[node_a].add(node_b)
                self.edges[node_b].add(node_a)
                edges_to_add -= 1

    def calculate_number_edges(self):
        num_edges = 0
        for n in self.edges:
            for n_2 in self.edges[n]:
                if n <= n_2:
                    num_edges += 1
        return num_edges

    def get_neighbors(self, node):
        return self.edges[node]

    def shortest_path_length(self, start, target):
        visited = set()
        node_queue = deque()
        node_queue.append((start, 0))
        while len(node_queue) > 0:
            next_node = node_queue.popleft()
            visited.add(next_node[0])
            if next_node[0] == target:
                return next_node[1]
            neighbors = self.get_neighbors(next_node[0])
            for n in neighbors:
                if n not in visited:
                    node_queue.append((n, next_node[1] + 1))
        return None

def save_graph(graph, path):
    with open(path, 'w') as f:
        num_nodes = len(graph.nodes)
        f.write(str(num_nodes) + '\n')
        for n in graph.nodes:
            f.write(str(n) + '\n')
        edges = []
        for a in graph.edges:
            for b in graph.edges[a]:
                if a <= b:
                    edges.append((a, b))
        num_edges = len(edges)
        f.write(str(num_edges) + '\n')
        for e in edges:
            a, b = e
            f.write(str(a) + ',' + str(b) + '\n')

def load_graph(path):
    with open(path, 'r') as f:
        num_nodes = int(f.readline().strip())
        nodes = []
        for _ in range(num_nodes):
            n = int(f.readline().strip())
            nodes.append(n)
        graph = Graph(nodes)
        num_edges = int(f.readline().strip())
        for _ in range(num_edges):
            edge_line = f.readline().strip()
            edges_str_arr = edge_line.split(',')
            a = int(edges_str_arr[0])
            b = int(edges_str_arr[1])
            graph.add_edge((a, b))
    return graph

def load_current_graph():
    graph_path = os.path.join(os.environ['DT_ROOT'], 'graph', 'sample_random_graph.txt')
    return load_graph(graph_path)

def create_graph():
    graph_path = os.path.join(os.environ['DT_ROOT'], 'graph', 'sample_random_graph.txt')
    num_vertices = 20
    edge_sparsity_coefficient = 0.1
    rand_graph = generate_random_connected_graph(num_vertices, edge_sparsity_coefficient, seed=1)
    save_graph(rand_graph, graph_path)

# https://stackoverflow.com/questions/2041517/random-simple-connected-graph-generation-with-given-sparseness
def generate_random_connected_graph(num_vertices, edge_sparsity_coefficient, seed=None):
    if seed is not None:
        random.seed(seed)

    nodes = list(range(num_vertices))

    # Create random minimum spanning tree
    S, T = set(nodes), set()

    curr_node = random.sample(S, 1).pop()
    S.remove(curr_node)
    T.add(curr_node)

    graph = Graph(nodes)

    while S:
        neighbor_node = random.sample(nodes, 1).pop()
        if neighbor_node not in T:
            edge = (curr_node, neighbor_node)
            graph.add_edge(edge)
            S.remove(neighbor_node)
            T.add(neighbor_node)
        curr_node = neighbor_node

    # Add remaining edges at random
    desired_edges = num_vertices ** 2 * edge_sparsity_coefficient
    graph.add_random_edges(desired_edges - num_vertices + 1)

    return graph

