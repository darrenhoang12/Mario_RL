import torch
import numpy as np
import random

from model import DDQN
from collections import deque


class Mario:
    def __init__(self, state_dim, action_dim, save_dir):
        """Mario reinforcement learning agent with DDQN."""
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = DDQN(self.state_dim, self.action_dim).float()
        self.model = self.model.to(device=self.device)

        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.step = 0
        
        self.save_every = 5e5

    def act(self, state):
        """Given input state, choose epsilon greedy action"""
        pass

    def cache(self, exp):
        """Add the experience to memory"""
        pass

    def recall(self):
        """Sample experiences from memory"""
        pass

    def learn(self):
        """Update online action value function with a batch of experiences"""
        pass