import random

class Graph:
    def __init__(self, nodes):
        self.edges = {}
        self.nodes = nodes
        for n in nodes:
            self.edges[n] = set()

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
            if node_b not in self.edges[node_a]:
                self.edges[node_a] = node_b
                self.edges[node_b] = node_a
                edges_to_add -= 1

# https://stackoverflow.com/questions/2041517/random-simple-connected-graph-generation-with-given-sparseness
def generate_random_connected_graph(num_vertices, edge_sparsity_coefficient):
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
    max_edges = (num_vertices - 1) * num_vertices / 2
    desired_edges = max_edges * edge_sparsity_coefficient
    num_edges_to_add = desired_edges - num_vertices + 1
    graph.add_random_edges(num_edges_to_add)

    return graph

