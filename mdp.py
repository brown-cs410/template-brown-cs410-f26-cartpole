import numpy as np
import gymnasium as gym
from typing import DefaultDict, List, Tuple, Dict, Optional

class MDP(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def reset(self):
        """
        Resets the environment to an initial state.
        """
        pass

    def step(self, action):
        """
        Takes an action from a state, and returns another state (occasionally with some other information)
        """
        pass

    def display_state(self):
        """
        Displays the current state of the game in text or some rendered form.
        """
        pass

class CartPole(gym.Wrapper):
    def __init__(self):
        """
        Initializes the CartPole environment from Gymnasium.
        """
        env = gym.make('CartPole-v1', render_mode='rgb_array')
        super().__init__(env)  
        self.current_state = self.env.reset()[0]
        self.action_space = [0, 1]
        
    @property
    def action_size(self):
        return self.unwrapped.action_space.n

    @property
    def features_size(self):
        return self.phi(self.unwrapped.observation_space.sample()).shape[0]

    @property
    def gamma(self):
        return 0.99
        
    def reset(self):
        """
        Resets the environment to start a new episode.
        Returns the initial state of the episode.
        """
        self.current_state = self.env.reset()[0]
        return self.current_state

    def step(self, action):
        """
        Takes an action in the CartPole environment.
        Action can be either 0 (move left) or 1 (move right).
        
        Returns:
            next_state: The state after taking the action.
            reward: Reward received after taking the action.
            done: Boolean indicating if the episode is over.
            info: Additional info about the current step.
        """
        next_state, reward, done, _, _ = self.env.step(action)
        self.current_state = next_state if not done else None
        return next_state, reward, done
    
    def display_state(self):
        """
        Displays the current state of the CartPole environment in a readable format.
        """
        print(f"Cart Position: {self.current_state[0]:.2f}, "
            f"Cart Velocity: {self.current_state[1]:.2f}, "
            f"Pole Angle: {self.current_state[2]:.2f}, "
            f"Pole Angular Velocity: {self.current_state[3]:.2f}")
    
    def play_automatically(self, max_steps=200):
        """
        Plays the CartPole environment automatically using a simple strategy.
        The cart will move right if the pole is leaning to the right, and left if the pole is leaning to the left.

        Parameters:
        - max_steps: Maximum number of steps to run the simulation for.
        """
        try:
            step_count = 0
            while step_count < max_steps:
                self.display_state()
                
                action = 0 if self.current_state[2] < 0 else 1
                next_state, reward, done, _ = self.step(action)
                
                action_str = 'Move Left' if action == 0 else 'Move Right'
                print(f"Action taken: {action_str}")

                if done:
                    self.display_state()
                    print(f"Episode finished. Total Reward: {reward}")
                    break
                
                step_count += 1
        except KeyboardInterrupt:
            print("Simulation interrupted.")
            
class FeaturizedCartPole(gym.Wrapper):
    def __init__(self, phi):
        """
        Initializes the CartPole environment from Gymnasium with a feature transformation function phi.
        """
        env = gym.make('CartPole-v1', render_mode='rgb_array')
        super().__init__(env)  
        self.phi = phi
        self.current_state = self.phi(self.env.reset()[0])
    
    @property
    def action_size(self):
        return self.unwrapped.action_space.n

    @property
    def features_size(self):
        return self.phi(self.unwrapped.observation_space.sample()).shape[0]

    @property
    def gamma(self):
        return 0.99
    
    def reset(self) -> Tuple[np.ndarray, dict]:
        """
        Resets the environment and returns the initial state as a feature vector.
        The reset function also returns any additional information in `flags_dict`,
        which is part of the Gym environment's reset protocol.

        Returns:
            A tuple containing the feature vector of the initial state and a dictionary of flags.
        """
        start_state, flags_dict = self.env.reset()
        self.current_state = self.phi(start_state)
        return self.current_state
    
    def step(self, action):
        """
        Takes a step in the environment using the given action. The environment
        transitions to the next state, which is then transformed into a feature
        vector. This function also returns the reward for the action, a boolean
        indicating if the episode has ended (`done`), and another boolean indicating
        if the episode was truncated (`truncation_flag`).

        Args:
            action (int): The action to take in the environment.

        Returns:
            A tuple containing:
            - The feature vector of the next state
            - The reward received for taking the action
            - A boolean indicating if the episode has ended (done)
            - A boolean indicating if the episode was truncated (truncation_flag)
        """
        next_state, reward, done, truncation_flag, flags_dict = self.env.step(action)
        self.current_state = self.phi(next_state) if not done else None
        return self.phi(next_state), reward, done, truncation_flag

    def display_state(self):
        """
        Displays the current state of the FeaturizedCartPole environment in a readable format.
        """
        for i in range(len(self.current_state)):
            print(f'Feature {i}: {self.current_state[i]:.2f}, ')