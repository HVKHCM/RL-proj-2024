import gym
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage, DummyVecEnv

from stable_baselines3.common.callbacks import BaseCallback

import torch

class RewardCallback(BaseCallback):
    """
    Custom callback for tracking the cumulative reward per episode and printing the final reward.
    """
    def __init__(self, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.cumulative_reward = 0.0
        self.episode_rewards = []

    def _on_step(self) -> bool:
        """
        This method is called in the training loop after each step.
        """
        self.cumulative_reward += self.locals['rewards'][0]  # Assuming single environment for simplicity
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.cumulative_reward)
            self.cumulative_reward = 0.0  # Reset for the next episode
        return True

    def _on_training_end(self):
        """
        This method is called at the end of training.
        """
        if self.episode_rewards:
            print(f"Final Reward: {self.episode_rewards[-1]}")
        else:
            print("No episodes completed.")


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Assuming the original observation space is an image, update its shape for grayscale
        obs_shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(obs_shape[0], obs_shape[1], 1), dtype=np.uint8)

    def observation(self, observation):
        # Convert observation to grayscale
        grayscale_obs = np.dot(observation[...,:3], [0.2989, 0.5870, 0.1140])
        # Ensure the grayscale observation has the expected shape (height, width, 1)
        grayscale_obs = np.expand_dims(grayscale_obs, axis=-1)
        return grayscale_obs.astype(np.uint8)

def make_env():
    env = gym.make("MsPacmanNoFrameskip-v4")
    env = GrayScaleObservation(env)
    return env

def predict_action(model, single_image, device='cpu'):
    """
    Predict an action based on a single input image for the A2C model.

    Parameters:
    - model: The trained A2C model.
    - single_image: A single input image (numpy array).
    - device: Computation device ('cpu' or 'cuda').

    Returns:
    - action: The predicted action.
    """
    # Ensure the image is in grayscale and has dimensions [H, W, C] where C=1
    if len(single_image.shape) == 2:  # If missing channel dimension
        single_image = np.expand_dims(single_image, axis=-1)
    elif single_image.shape[2] > 1:  # If RGB, convert to grayscale
        single_image = np.dot(single_image[..., :3], [0.2989, 0.5870, 0.1140]).reshape(single_image.shape[0], single_image.shape[1], 1)
    
    # Normalize the image
    single_image = single_image / 255.0

    # Stack the image to create a 'batch' [B, C, H, W] with B=1 for a single observation
    image_stack = np.repeat(single_image[np.newaxis, :], 4, axis=3)  # Mock stack 4 frames
    image_stack = np.transpose(image_stack, (0, 3, 1, 2))  # Transpose to [B, C, H, W]

    # Directly pass the numpy array; conversion to tensor and device placement will be handled internally
    action, _ = model.predict(image_stack, deterministic=True)
    
    return action

# Example usage:
# Assuming 'input_image' is your preprocessed image ready to be input:
# action = predict_action(model, input_image)
# print("Predicted Action:", action)


# Create the environment, apply custom wrappers, and then use DummyVecEnv for vectorization
env = DummyVecEnv([make_env])
# Stack frames to give the model a sense of motion
env = VecFrameStack(env, n_stack=4)
# Use VecTransposeImage to ensure the observation is channel-first, which is required by PyTorch
env = VecTransposeImage(env)

# Instantiate the callback
reward_callback = RewardCallback()

# Initialize the model with PPO and a CNN policy suitable for handling image observations
model = A2C("CnnPolicy", env, verbose=1)

# Train the model; consider increasing total_timesteps for better performance
model.learn(total_timesteps=int(1e6), callback = reward_callback)

#Save model
model_path = "a2c-1000000.zip"  # Choose your path and file name
model.save(model_path)