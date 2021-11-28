import os, sys
sys.path.append(os.environ['DT_ROOT'])

import torch
from torch.nn import functional as F
import numpy as np

from graph_dataset import GraphDataset
from dt import DecisionTransformer
import constants
import train
from eval import evaluate_model

batch_size = 20
num_epochs = 20
eval_interval = 1
learning_rate = 1e-3
decay_rate = 1e-4

device = torch.device('cuda')

train_dataset = GraphDataset()
train_size = len(train_dataset)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=10, drop_last=True)

emb_sz = 256
timesteps = 10
nhead = 1
nlayers = 1
model = DecisionTransformer(emb_sz, timesteps, nhead, nlayers, device)
model = model.to(device)

# Train on all actions except last
def criterion(outputs, labels):
    curr_batch_size = outputs.shape[0]
    seq_length = outputs.shape[1]

    output_actions = outputs[:,:-1,:]
    gt_actions = labels[:,:-1].long()
    output_actions_squashed = output_actions.reshape(curr_batch_size * (seq_length - 1), constants.TOTAL)
    gt_actions_squashed = gt_actions.reshape(curr_batch_size * (seq_length - 1))

    return F.cross_entropy(output_actions_squashed, gt_actions_squashed, reduction='sum')

def inputs_labels_func(data):
    rewards, states, actions = data
    seq_length = rewards.shape[1]
    curr_batch_size = rewards.shape[0]

    t = np.arange(seq_length, dtype=np.int32)
    t = np.repeat(t[None, :], curr_batch_size, axis=0)

    rewards = torch.tensor(rewards).to(device)
    states = torch.tensor(states).to(device)
    actions = torch.tensor(actions).to(device)
    t = torch.tensor(t).to(device)
    return (rewards, states, actions, t), actions

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=decay_rate
)

train.train(train_dataloader, train_size, inputs_labels_func,
            model, criterion, optimizer,
            device=device, num_epochs=num_epochs, do_carriage_return=True,
            eval_func=evaluate_model, eval_interval=eval_interval)