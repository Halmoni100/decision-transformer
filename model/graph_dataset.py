import os
import numpy as np

from torch.utils.data import Dataset

class GraphDataset(Dataset):
    def __init__(self):
        data_dir = os.path.join(os.environ['DT_ROOT'], 'data')
        self.rewards = np.load(os.path.join(data_dir, 'rewards.npy'))
        self.states = np.load(os.path.join(data_dir, 'states.npy'))
        self.actions = np.load(os.path.join(data_dir, 'actions.npy'))

    def __getitem__(self, index):
        return self.rewards[index], self.states[index], self.actions[index]

    def __len__(self):
        return self.rewards.shape[0]