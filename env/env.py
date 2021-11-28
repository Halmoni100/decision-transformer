import constants
from graph import graph_util

class Env:
    def __init__(self):
        self.graph = graph_util.load_current_graph()
        self.goal = constants.GOAL

        self.curr_node = None

    def reset(self, curr_node):
        self.curr_node = curr_node

    # Return (state, reward)
    def step(self, action):
        if action in self.graph.get_neighbors(self.curr_node):
            self.curr_node = action
        return self.curr_node

    def get_reward(self):
        if self.curr_node == self.goal:
            return 0
        else:
            return -1
