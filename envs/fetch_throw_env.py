import types

import gymnasium as gym
import numpy as np

# Default Cartesian motion scale applied inside Fetch's `_set_action` (see README Notes).
# Actions stay in [-1, 1]; this multiplies the internal 0.05 m/step mocap delta. Use the same
# value for data collection, BC, and RL so the MDP matches.
DEFAULT_THROW_OVERCLOCK_FACTOR = 3.0


def _patch_fetch_pos_scale(unwrapped_env, factor: float) -> None:
    """Scale Fetch Cartesian mocap deltas beyond the default 0.05 m/step cap.

    Gymnasium Fetch applies `pos_ctrl *= 0.05` inside `_set_action` (see
    `gymnasium_robotics.envs.fetch.fetch_env.BaseFetchEnv._set_action`).
    The env also clips actions to [-1, 1] in `BaseRobotEnv.step`, so multiplying
    actions *before* `step()` cannot increase motion — it gets clipped away.

    This patch multiplies that internal 0.05 scale by `factor` on the unwrapped
    env instance so overclocking actually affects physics.
    """
    factor = float(max(1.0, factor))
    unwrapped_env._throw_pos_scale = factor
    if factor <= 1.0:
        return

    def _set_action_scaled(self, action):
        assert action.shape == (4,)
        action = action.copy()
        pos_ctrl, gripper_ctrl = action[:3], action[3]
        pos_ctrl *= 0.05 * getattr(self, "_throw_pos_scale", 1.0)
        rot_ctrl = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float64)
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl], dtype=np.float64)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        full_action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # New mujoco bindings (Fetch v2+)
        if hasattr(self, "model") and hasattr(self, "data"):
            self._utils.ctrl_set_action(self.model, self.data, full_action)
            self._utils.mocap_set_action(self.model, self.data, full_action)
        # mujoco_py (legacy)
        elif hasattr(self, "sim"):
            self._utils.ctrl_set_action(self.sim, full_action)
            self._utils.mocap_set_action(self.sim, full_action)
        else:
            raise RuntimeError("Unsupported Fetch backend: expected model/data or sim.")

    unwrapped_env._set_action = types.MethodType(_set_action_scaled, unwrapped_env)


class FetchThrowWrapper(gym.Wrapper):
    def __init__(self, env, throw_overclock_factor=None):
        super().__init__(env)
        self.has_scored = False
        if throw_overclock_factor is None:
            throw_overclock_factor = DEFAULT_THROW_OVERCLOCK_FACTOR
        self.throw_overclock_factor = float(max(1.0, throw_overclock_factor))
        _patch_fetch_pos_scale(self.env.unwrapped, self.throw_overclock_factor)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.has_scored = False 
        
        hoop_center = np.array([2.595, 0.75, 0.7])
        
        obs['desired_goal'] = hoop_center.copy()
        self.env.unwrapped.goal = hoop_center.copy()
        
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        ball_pos = obs['achieved_goal'] 
        
        # Updated bounding box for the new hoop location
        in_x = 2.5 <= ball_pos[0] <= 2.7
        in_y = 0.65 <= ball_pos[1] <= 0.85
        
        # New Z-bounds: The rim is at 0.7. Catch the ball as it falls from 0.8 down to 0.4.
        in_z = 0.4 <= ball_pos[2] <= 0.8
        
        if in_x and in_y and in_z:
            self.has_scored = True
            
        if self.has_scored:
            info['is_success'] = 1.0
            reward = 0.0 
            
        return obs, reward, terminated, truncated, info
