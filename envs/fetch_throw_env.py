import gymnasium as gym
import numpy as np

class FetchThrowWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # The base fetch arm has a reach of roughly ~0.8 to 1.0 meters from its base.
        # We will override the goal generation to force it further away.
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Override the goal to be out of reach (e.g., further along the X or Y axis)
        # This is a hacky but effective way to force a throw.
        new_goal = obs['achieved_goal'].copy()
        new_goal[0] += 1.5  # Push goal 1.5 meters away on the X axis
        new_goal[2] = 0.0   # Goal on the table/floor
        
        obs['desired_goal'] = new_goal
        self.env.unwrapped.goal = new_goal
        return obs, info