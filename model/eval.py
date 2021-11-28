import os, sys
sys.path.append(os.environ['DT_ROOT'])

import numpy as np
import torch

from env.env import Env
import constants

def evaluate_model(model, inputs_labels_func):
    goal_reached = 0

    env = Env()

    for n in range(constants.TOTAL):
        if n == constants.GOAL:
            continue
        rewards_to_go = np.zeros((1, 10), dtype=np.float32)
        states = np.zeros((1, 10), dtype=np.int32)
        actions = np.zeros((1, 10), dtype=np.int32)
        env.reset(n)
        shortest_path_len = env.graph.shortest_path_length(n, constants.GOAL)
        rewards_to_go[0][0] = shortest_path_len
        states[0][0] = n
        actions[0][0] = constants.GOAL

        for i in range(shortest_path_len):
            model_input, _ = inputs_labels_func((rewards_to_go[:,:i+1], states[:,:i+1], actions[:,:i+1]))
            output_actions = model(model_input)
            last_action = float(torch.argmax(output_actions[0][-1]).cpu())
            reward_curr_state = env.get_reward()
            next_state = env.step(last_action)
            states[0][i+1] = next_state
            rewards_to_go[0][i+1] = rewards_to_go[0][i] - reward_curr_state
            actions[0][i] = last_action
            actions[0][i+1] = constants.GOAL

        if next_state == constants.GOAL:
            goal_reached += 1

    print("Goal reached: {}/{}".format(goal_reached, constants.TOTAL))
