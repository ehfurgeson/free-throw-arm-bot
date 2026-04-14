import gymnasium as gym
import numpy as np
from expert_policy import compute_expert_action
from envs.fetch_throw_env import FetchThrowWrapper

def collect_data(num_successes_needed=100):
    base_env = gym.make('FetchPickAndPlace-v2') 
    env = FetchThrowWrapper(base_env)

    successful_trajectories = []
    successes_collected = 0
    
    while successes_collected < num_successes_needed:
        obs, info = env.reset()
        done = False
        truncated = False
        
        current_trajectory = {'obs': [], 'actions': []}
        
        while not (done or truncated):
            # 1. Ask the expert from Step 1.3 what to do
            action = compute_expert_action(obs)
            
            # 2. Save the state and the action
            current_trajectory['obs'].append(obs['observation'])
            current_trajectory['actions'].append(action)
            
            # 3. Take the step in the environment
            obs, reward, done, truncated, info = env.step(action)
            
        # 4. Check if the throw was actually a success
        if info.get('is_success', 0.0) == 1.0:
            successful_trajectories.append(current_trajectory)
            successes_collected += 1
            print(f"Success! Collected {successes_collected}/{num_successes_needed}")
            
    # 5. Save out to a file format your BC script can read
    np.savez('data/expert_demonstrations.npz', trajectories=successful_trajectories)

if __name__ == "__main__":
    collect_data()