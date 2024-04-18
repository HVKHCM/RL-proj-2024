import gym
import numpy as np
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.atari_wrappers import AtariWrapper
from gym.wrappers import GrayScaleObservation, ResizeObservation

def make_grayscale_env(env_id, n_envs=1, seed=0):
    """
    Create an environment with grayscale conversion.
    """
    def make_env():
        env = gym.make(env_id)
        env = AtariWrapper(env)  # Apply standard Atari preprocessing
        env = GrayScaleObservation(env)  # Convert frames to grayscale
        env = ResizeObservation(env, (84, 84))  # Resize frames to 84x84
        env.seed(seed)
        return env

    env = DummyVecEnv([make_env for _ in range(n_envs)])
    env = VecFrameStack(env, n_stack=4)  # Stack 4 frames
    return env