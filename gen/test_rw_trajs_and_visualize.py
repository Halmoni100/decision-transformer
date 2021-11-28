import os, sys
sys.path.append(os.environ['DT_ROOT'])

import os
import numpy as np
import matplotlib.pyplot as plt

from env.env import Env
import constants

data_dir = os.path.join(os.environ['DT_ROOT'], 'data')
rewards = np.load(os.path.join(data_dir, 'rewards.npy'))
states = np.load(os.path.join(data_dir, 'states.npy'))
actions = np.load(os.path.join(data_dir, 'actions.npy'))

env = Env()

num_trajs = rewards.shape[0]
traj_length = rewards.shape[1]
path_to_goal_lengths = []
nodes_with_shortest_path = set()
for traj in range(num_trajs):
    curr_rewards_to_go = rewards[traj][0]
    curr_state = states[traj][0]
    env.reset(curr_state)
    goal_reach_t = -1
    for i in range(traj_length-1):
        shortest_path = env.graph.shortest_path_length(curr_state, constants.GOAL)
        if shortest_path > 0 and shortest_path == -curr_rewards_to_go:
            nodes_with_shortest_path.add(curr_state)
        next_rewards_to_go = curr_rewards_to_go - env.get_reward()
        next_state = env.step(actions[traj][i])
        assert(states[traj][i+1] == next_state)
        assert(rewards[traj][i+1] == next_rewards_to_go)
        if goal_reach_t == -1 and next_state == constants.GOAL:
            goal_reach_t = i + 1
            assert(-rewards[traj][0] == goal_reach_t)
        if curr_state == constants.GOAL:
            assert(next_state == curr_state)
            assert(actions[traj][i] == curr_state)
            assert(rewards[traj][i] == 0)
            assert(rewards[traj][i+1] == 0)
        curr_rewards_to_go = next_rewards_to_go
        curr_state = next_state
    path_to_goal_lengths.append(goal_reach_t)
    if goal_reach_t == -1:
        for i in range(traj_length):
            assert(rewards[traj][i] == np.NINF)

print(len(nodes_with_shortest_path))

plt.hist(path_to_goal_lengths)
plt.show()

