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

        self.memory = deque(maxlen=100000)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = DDQN(self.state_dim, self.action_dim).float()
        self.model = self.model.to(device=self.device)

        self.batch_size = 32

        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.step = 0
        
        self.save_every = 5e5

    def act(self, state):
        """
        Given input state, choose epsilon greedy action
        
        Inputs:
            state: A single frame from the preprocessed mario environment
        
        Returns:
            action_idx: The index of the chosen action (SIMPLE_MOVEMENT)
        """
        # Exploring (random) vs exploiting (model)
        if np.random.rand() < self.epsilon:
            action_idx = np.random.randint(self.action_dim)
        else:
            state_t = torch.tensor(state, device=self.device)
            action_values = self.model(state_t, model='online')
            action_idx = torch.argmax(action_values).item()

        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        self.step += 1

        return action_idx

    def cache(self, state, next_state, action, reward, done):
        """
        Add the experience to memory
        
        Inputs:
            state: The original observation from the preprocessed mario environment
            next_state: The next observation from the preprocessed mario environment after the action is performed by the agent
            action: Action performed by agent
            reward: Reward received after the action
            done: Whether the episode is over
        """

        state_t = torch.tensor(state, device=self.device)
        next_state_t = torch.tensor(next_state, device=self.device)
        action_t = torch.tensor([action], device=self.device)
        reward_t = torch.tensor([reward], device=self.device)
        done_t = torch.tensor([done], device=self.device)

        self.memory.append((state_t, next_state_t, action_t, reward_t, done_t))

    def recall(self):
        """
        Sample batch of experiences from memory
        """
        batch = random.sample(self.memory, self.batch_size)
        # batch is 32 random tuples sampled from the self.memory deque and each s in the form (state, next_state, action, reward, done)
        states, next_states, actions, rewards, dones = map(torch.stack, zip(*batch))
        return states, next_states, actions.squeeze(), rewards.squeeze(), dones.squeeze()


    def learn(self):
        """Update online action value function with a batch of experiences"""
        pass