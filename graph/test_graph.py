import graph_util
from collections import deque

rand_graph = graph_util.load_current_graph()
num_vertices = 20
edge_sparsity_coefficient = 0.1
num_edges = rand_graph.calculate_number_edges()
assert(num_edges == num_vertices ** 2 * edge_sparsity_coefficient)

# Check if fully connected
visited = set()
node_queue = deque()
curr_node = 0
node_queue.append(curr_node)
while len(node_queue) > 0:
    next_node = node_queue.popleft()
    neighbors = rand_graph.get_neighbors(next_node)
    for n in neighbors:
        if n not in visited:
            node_queue.append(n)
            visited.add(n)
assert(len(visited) == 20)