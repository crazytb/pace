import random
import numpy as np
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'cum_reward', 'tau', 'done'))

class ReplayMemory():
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)
    
class DQN(nn.Module):
    def __init__(self, n_observations=4, n_actions=2, history_length=10):
        super(DQN, self).__init__()
        self.n_observations = n_observations or 4  # Default to 4 for backward compatibility
        
        # Simple network for current 4-dimensional observation
        # Input: [obss_remain, radio_transition_time, tx_duration, cw_index]
        self.layer1 = nn.Linear(self.n_observations, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, n_actions)
        
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """
        Process simple 4-dimensional observation vector
        Input: [obss_remain, radio_transition_time, tx_duration, cw_index]
        """
        # Simple feedforward network
        x = F.relu(self.layer1(x))
        x = self.dropout(x)
        x = F.relu(self.layer2(x))
        return self.layer3(x)
