import random, datetime
from pathlib import Path

import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from nes_py.wrappers import JoypadSpace

from agent import Mario
from wrappers import ResizeObservation, SkipFrame


def replay():
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')

    env = JoypadSpace(
                    env,
                    [['right'],
                    ['right', 'A']]
                    )

    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, shape=84)
    env = TransformObservation(env, f=lambda x: x / 255.)
    env = FrameStack(env, num_stack=4)

    env.reset()

    save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    save_dir.mkdir(parents=True)

    checkpoint = Path('checkpoints/2023-07-28T21-56-58/mario_net_4.chkpt')
    mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir, checkpoint=checkpoint)
    mario.epsilon = mario.epsilon_min

    episodes = 100

    for e in range(episodes):

        state = env.reset()

        while True:

            env.render()

            action = mario.act(state)

            next_state, reward, done, info = env.step(action)

            mario.cache(state, next_state, action, reward, done)

            state = next_state

            if done or info['flag_get']:
                break

if __name__ == '__main__':
    replay()