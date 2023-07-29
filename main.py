import os
import wandb
import torch
from PIL import Image
from torch import nn
from torchvision import transforms as T
from PIL import Image
from pathlib import Path
import numpy as np
import datetime

from gym.wrappers import FrameStack, GrayScaleObservation
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from wrappers import SkipFrame, ResizeObservation
from agent import Mario
import matplotlib.pyplot as plt

def main():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    # Apply preprocessing
    env = SkipFrame(env, 4)
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, 4)

    save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)
    episodes = 40000
    # TODO: wandb logging
    for e in range(episodes):
        state = env.reset()
        while True:
            action = mario.act(state)
            next_state, reward, done, info = env.step(action)
            mario.cache(state, next_state, action, reward, done)
            q, loss = mario.learn()
            state = next_state
            if done or info['flag_get']:
                break


if __name__ == '__main__':
    main()


