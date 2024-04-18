

if reward_callback.episode_rewards:
    print(f"Final Reward from outside the callback: {reward_callback.episode_rewards[-1]}")
else:
    print("No episodes completed.")


# Generate a mock grayscale image of size (210, 160, 1)
# We'll fill it with random values to simulate an actual game frame.
# In a real scenario, this would be a preprocessed frame from the environment.
mock_image = np.random.randint(0, 256, (210, 160, 1), dtype=np.uint8)

# Assuming `model` is your trained A2C model and `predict_action` is defined,
# let's predict the action for this mock image.
action = predict_action(model, mock_image)

print("Predicted Action:", action)