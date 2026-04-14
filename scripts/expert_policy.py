import numpy as np

def compute_expert_action(obs):
    # This is a simplified pseudo-heuristic. Will need to tune these thresholds.
    gripper_pos = obs['observation'][:3]
    object_pos = obs['observation'][3:6]
    object_rel_pos = obs['observation'][6:9]
    
    # 1. Hover and Grasp
    if np.linalg.norm(object_rel_pos) > 0.05:
        # Move towards object
        action = np.append(object_rel_pos * 10, [-1.0]) # Keep gripper open (-1)
    elif obs['observation'][20] > 0.01: # Check if fingers are closed
        # Close gripper (1.0)
        action = np.array([0, 0, 0, 1.0])
    # 2. Wind-up
    elif gripper_pos[0] > 1.0: # Arbitrary back position
        action = np.array([-1.0, 0, -0.5, 1.0]) # Move back and down
    # 3. Whip and Release
    else:
        # Move forward and up fast
        action = np.array([1.0, 0, 1.0, 1.0]) 
        # Release trigger based on velocity or position threshold
        if gripper_pos[0] > 1.3: # Release point
            action[3] = -1.0 # Open gripper
            
    # Clip actions to valid space [-1, 1]
    return np.clip(action, -1.0, 1.0)