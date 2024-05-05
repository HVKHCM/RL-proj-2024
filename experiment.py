import gym
import numpy as np
import cv2
from gym.spaces import Box
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.policies import NatureCNN
import matplotlib.pyplot as plt
import random
import time


def reset_with_random_seed(env):
    random_seed = int(time.time()) + random.randint(0, 1000)
    env.seed(random_seed)
    return env.reset()

def randomize_state(env, num_random_steps=30):
    """ Perform random actions to reach a random initial state """
    obs = env.reset()
    for _ in range(num_random_steps):
        action = [env.action_space.sample()]  # Ensure it's a list for VecEnv
        obs, _, done, _ = env.step(action)
        if done[0]:  # Check done state within a list
            obs = env.reset()
    return obs

class GrayscaleNoisyInputWrapper(gym.ObservationWrapper):
    def __init__(self, env, noise_level=0.0):
        super().__init__(env)
        self.noise_level = noise_level
        # Update the observation space to reflect grayscale images
        obs_shape = self.observation_space.shape[:2] + (1,)  # Assumes original obs space is HxWx3
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        # Convert observation to grayscale and normalize to range 0-1
        grayscale_obs = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        grayscale_obs = np.expand_dims(grayscale_obs, axis=-1)  # Add channel dimension
        # Add noise
        noise = np.random.normal(0, self.noise_level, grayscale_obs.shape)
        noisy_obs = np.clip(grayscale_obs + noise, 0, 255).astype(np.uint8)
        return noisy_obs

    def set_noise_level(self, noise_level):
        self.noise_level = noise_level

env_id = "BreakoutNoFrameskip-v4"
env = gym.make(env_id)
env = GrayscaleNoisyInputWrapper(env, noise_level=0.00)  # Adjust noise level if needed
env = DummyVecEnv([lambda: env])  # Wrap with DummyVecEnv
env = VecTransposeImage(env)  # Ensure the observation order is (height, width, channel)

""" model = A2C("CnnPolicy", env, verbose=1, policy_kwargs={
    "features_extractor_class": NatureCNN,
    "features_extractor_kwargs": {"features_dim": 512}
}) 

model = PPO("CnnPolicy", env, verbose=1, policy_kwargs={
    "features_extractor_class": NatureCNN,
    "features_extractor_kwargs": {"features_dim": 512}
})

model = DQN("CnnPolicy", env, verbose=1, policy_kwargs={
    "features_extractor_class": NatureCNN,
    "features_extractor_kwargs": {"features_dim": 512}
}) 

model.learn(total_timesteps=int(1e8))"""


""" 
### Bruteforce targetted attack
check_point = range(0,1,0.001)
stop_point = []
for i in range(1000):
    action1 = [0]
    action2 = [0]
    while (action1[0] != action2[0]):
        obs1 = randomize_state(env,100)
        obs2 = randomize_state(env,100)
        action1, _state = model.predict(obs1, deterministic=True)
        action2, _state = model.predict(obs2, deterministic=True)
    noise = np.squezze(obs2) - np.squeeze(obs1)
    
    for j in check_point:
        new_obs = np.squeeze(obs1) + j*noise
        new_obs = np.expand_dims(new_obs,axis=0)
        new_obs = np.expand_dims(new_obs,axis=0)
        tmp_act, _state = model.predict(new_obs, deterministic=True)
        if tmp_act[0] != action1[0]:
            obj = (obs1, obs2, j, action1, action2, tmp_act, noise)
            stop_point.append(obj)
            break """

""" # Manipulate a single input and observe model reaction to gradual noise increase, plot the image
fig, ax = plt.subplots(1, 4, figsize=(20, 8))  # Adjusted for better layout
ax = ax.flatten()

obs1 = stop_point[473][0]
obs2 = stop_point[473][1]
new_obs = np.squeeze(obs1) + stop_point[473][2]*stop_point[6]
next_point = np.squeeze(new_obs) + 0.001*stop_point[6]
new_obs = np.expand_dims(np.expand_dims(new_obs), axis=0)
next_obs = np.expand_dims(np.expand_dims(next_point), axis=0)


ax[0].imshow(np.squeeze(obs1),cmap='gray')
ax[1].imshow(np.squeeze(obs2),cmap='gray')
ax[2].imshow(np.squeeze(new_obs), cmap='gray')
ax[3].imshow(np.squeeze(next_obs), cmap='gray')
plt.show() """
""" 
### Natural Filter noise:
### Extreme case: where all noise filter are guranteed to be completed
### Phi changed depend on how noise generator was obtained
check_point = range(0,1,0.001)
stop_point = []
for i in range(1000):
    obs = randomize_state(env)
    ### These are some that we already tested, feel free to change to some more complicated noise
    generator = cv2.blur(np.squeeze(obs),(199,19))
    #generator = cv2.GaussianBlur(np.squeeze(obs),(199,199), 200)
    #generator = cv2.medianBlur(np.squeeze(obs), 20)
    noise = generator - np.squeeze(obs)
    action, _state = model.predict(obs, deterministic=True)
    for j in check_point:
        new_obs = np.squeeze(obs) + j*noise
        new_obs = np.expand_dims(new_obs,axis=0)
        new_obs = np.expand_dims(new_obs,axis=0)
        tmp_act, _state = model.predict(new_obs, deterministic=True)
        if tmp_act[0] != action1[0]:
            obj = (obs, j, action, tmp_act, noise)
            stop_point.append(obj)
            break """


""" # Manipulate a single input and observe model reaction to gradual noise increase, plot the image
fig, ax = plt.subplots(1, 3, figsize=(20, 8))  # Adjusted for better layout
ax = ax.flatten()

obs = stop_point[473][0]
new_obs = np.squeeze(obs) + stop_point[473][1]*stop_point[4]
next_point = np.squeeze(new_obs) + 0.001*stop_point[4]
new_obs = np.expand_dims(np.expand_dims(new_obs), axis=0)
next_obs = np.expand_dims(np.expand_dims(next_point), axis=0)


ax[0].imshow(np.squeeze(obs),cmap='gray')
ax[1].imshow(np.squeeze(new_obs), cmap='gray')
ax[2].imshow(np.squeeze(next_obs), cmap='gray')
plt.show() """


"""
### Future work: On a basic level, this work. However, it take a while to run
### as well as occasionally have error (about 30% of the time)
### Resource outage related error mostly
### Another alternative way is using binary search. However, this assume that
### a solution is actually existed (no theoretical foundation)
### we included the prototype of the alternative here anyway and further research might resolve these problem 
### (only tested for 5 states so far)

### Binary search
def phi_search(start, end, model, original, noise, epsilon):
	mid = (start + end)/2
	adv = np.squeeze(original) + mid*noise
    adv_obs = np.expand_dim(np.expand_dim(adv, axis=0),axis=0)
	adv_act, _state = model.predict(adv_obs, deterministic=True)
	if (mid - start < epsilon):
		return start
    orig_act, _state = model.predict(original, deterministic=True)
	if (adv_act[0] == orig_act[0]):
		max_phi = phi_search(mid, end, model, original, noise, epsilon)
	else:
		max_phi = phi_search(start, mid, model, original, noise, epsilon)
	return max_phi

###Example usage
orignal_obs = randomize_state(env, 100)
generator = cv2.blur(np.squeeze(obs),(199,19))
noise = generator - np.squeeze(obs)
phi = phi_search(0.0,1.0,model,original_obs,noise,0.001)
print(phi)
"""
