import os, sys
sys.path.append(os.environ['DT_ROOT'])

from graph import graph_util
import numpy as np
import random

from env.env import Env
import constants
from util import data_prep

random.seed(4)

num_trajs = 1000
traj_length = 10
rewards = np.zeros((num_trajs, traj_length), dtype=np.float32)
states = np.zeros((num_trajs, traj_length), dtype=np.int32)
actions = np.zeros((num_trajs, traj_length), dtype=np.int32)

env = Env()

num_goal_reached = 0
states_included = set()
states_started = set()

for i in range(num_trajs):
    start_node = random.randint(0, 19)
    while start_node == constants.GOAL:
        start_node = random.randint(0, 19)
    env.reset(start_node)
    states[i][0] = start_node
    curr_node = start_node
    goal_reach_t = -1
    for j in range(traj_length-1):
        if curr_node == constants.GOAL:
            curr_action = constants.GOAL
        else:
            neighbors = list(env.graph.get_neighbors(curr_node))
            rand_idx = random.randint(0, len(neighbors) - 1)
            curr_action = neighbors[rand_idx] # random.randint(0, 19)
        actions[i][j] = curr_action
        next_node = env.step(curr_action)
        states[i][j+1] = next_node
        if goal_reach_t == -1 and next_node == constants.GOAL:
            goal_reach_t = j+1
        curr_node = next_node
    actions[i][traj_length-1] = constants.GOAL
    if goal_reach_t != -1:
        num_goal_reached += 1
        for s in states[i]:
            states_included.add(s)
        states_started.add(states[i][0])
        reward_to_go = -goal_reach_t
        k = 0
        while reward_to_go < 0:
            rewards[i][k] = reward_to_go
            reward_to_go += 1
            k += 1
    else:
        for k in range(traj_length):
            rewards[i][k] = np.NINF

print(num_goal_reached)
print(len(states_included))
print(len(states_started))
data_dir = os.path.join(os.environ['DT_ROOT'], 'data')
data_prep.rm_and_mkdir(data_dir)
np.save(os.path.join(data_dir, 'rewards.npy'), rewards)
np.save(os.path.join(data_dir, 'states.npy'), states)
np.save(os.path.join(data_dir, 'actions.npy'), actions)