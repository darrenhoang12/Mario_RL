import torch
from torch import nn
from copy import deepcopy

class DDQNConv(nn.Module):
    """
    CNN Structure that Implements Double Q-learning
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c = input_dim[0]

        conv1 = nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4)
        conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        fc1 = nn.Linear(3136, 512)
        fc2 = nn.Linear(512, output_dim)

        self.online = nn.Sequential(
            conv1,
            nn.ReLU(),
            conv2,
            nn.ReLU(),
            conv3,
            nn.ReLU(),
            nn.Flatten(),
            fc1,
            nn.ReLU(),
            fc2
        )

        self.target = deepcopy(self.online)
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, input, model):
        if model == 'online':
            return self.online(input)
        elif model == 'target':
            return self.target(input)