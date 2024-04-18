from model import *
import torch.optim as optim

env = gym.make("MsPacman-v0")
input_shape = (1, 84, 84)  # Channel, Height, Width
n_actions = env.action_space.n

model = A2C(input_shape, n_actions)
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_episodes = 10

train_a2c(env, model, optimizer, num_episodes)