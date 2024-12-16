import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation

class OT2Env(gym.Env):
    def __init__(self, render=False, max_steps=1000):
        super(OT2Env, self).__init__()
        self.render = render
        self.max_steps = max_steps

        # Create the simulation environment
        self.sim = Simulation(num_agents=1)

        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        # keep track of the number of steps
        self.steps = 0

    def reset(self, seed=None):
        # being able to set a seed is required for reproducibility
        if seed is not None:
            np.random.seed(seed)

        # Reset the state of the environment to an initial state
        # set a random goal position for the agent, consisting of x, y, and z coordinates within the working area (you determined these values in the previous datalab task)
        self.goal_position = np.array([
            np.random.uniform(-0.1875, 0.2535),  # Random x within range
            np.random.uniform(-0.1705, 0.2199),  # Random y within range
            np.random.uniform(0.12, 0.2895)   # Random z within range
        ])
        # Call the environment reset function
        observation = self.sim.reset(num_agents=1)

        # Converting the pipette position into an numpy array
        pipette_position = np.array(observation[f'robotId_{self.sim.robotIds[0]}']['pipette_position'], dtype=np.float32)

        # Append the goal position to the pipette position
        observation = np.concatenate([pipette_position, self.goal_position], axis=0).astype(np.float32)

        # Reset the number of steps
        self.steps = 0

        info = {}

        return observation, info

    def step(self, action):
        # Ensure pipette_action is a NumPy array
        pipette_action = np.array(action, dtype=np.float32)

        # Append 0 for the drop action
        drop_action = np.array([0], dtype=np.float32)

        # Combine pipette action and drop action
        action = np.concatenate([pipette_action, drop_action])
        
        # Call the environment step function
        observation = self.sim.run([action]) # Why do we need to pass the action as a list? Think about the simulation class.

        # Converting the pipette position into an numpy array
        pipette_position = np.array(observation[f'robotId_{self.sim.robotIds[0]}']['pipette_position'], dtype=np.float32)

        # Append the goal position to the pipette position
        observation = np.concatenate([pipette_position, self.goal_position], axis=0).astype(np.float32)

        # Calculate the current distance to the goal
        current_distance = np.linalg.norm(pipette_position - self.goal_position)

        # Initialize previous_distance if it doesn't exist
        if not hasattr(self, 'previous_distance'):
            self.previous_distance = current_distance

        # Calculate the change in distance
        distance_change = self.previous_distance - current_distance

        # Define thresholds and their corresponding reward multipliers
        thresholds = [
            (0.03, 300),  # Highest reward multiplier
            (0.05, 200),  # Higher reward multiplier
            (0.15, 100),  # Normal reward multiplier
        ]

        # Determine the reward multiplier based on the thresholds
        reward_multiplier = 50  # Default multiplier for far distances
        for threshold, multiplier in thresholds:
            if current_distance <= threshold:
                reward_multiplier = multiplier
                break  # Use the first applicable multiplier (smallest threshold)

        # Assign reward based on movement direction and distance to the goal
        if distance_change > 0:  # Moving closer
            reward = reward_multiplier * distance_change  # Reward for moving closer
        else:  # Moving away
            reward = (reward_multiplier * 2) * distance_change  # Penalize more for moving away

        # Update the previous distance
        self.previous_distance = current_distance

        # What is a reasonable threshold? Think about the size of the pipette tip and the size of the plants.
        if current_distance < 0.1:
            terminated = True
        else:
            terminated = False

        # next we need to check if the episode should be truncated, we can check if the current number of steps is greater than the maximum number of steps
        if self.steps >= self.max_steps:
            truncated = True
        else:
            truncated = False

        info = {} # we don't need to return any additional information

        # increment the number of steps
        self.steps += 1

        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        pass
    
    def close(self):
        self.sim.close()
