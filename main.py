import torch
from PIL import Image
from torch import nn
from torchvision import transforms as T
from PIL import Image
import numpy as np

from gym.wrappers import FrameStack, GrayScaleObservation
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from wrappers import SkipFrame, ResizeObservation
import matplotlib.pyplot as plt

def main():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    # Apply preprocessing
    env = SkipFrame(env, 4)
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, 4)
    state = env.reset()
    print(state.shape)

if __name__ == '__main__':
    main()


