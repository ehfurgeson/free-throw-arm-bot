import types

import gymnasium as gym
import numpy as np

# Default Cartesian motion scale applied inside Fetch's `_set_action` (see README Notes).
# Actions stay in [-1, 1]; this multiplies the internal 0.05 m/step mocap delta. Use the same
# value for data collection, BC, and RL so the MDP matches.
DEFAULT_THROW_OVERCLOCK_FACTOR = 3.0

# Canonical fixed ball spawn position used when `fixed_object_position` is set.
# Centered roughly on the typical Fetch table sample range, sitting on the
# table surface (ball radius ~0.025 m, table top z ~0.4 m).
DEFAULT_FIXED_OBJECT_POSITION = (1.3, 0.75, 0.45)


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
    def __init__(
        self,
        env,
        throw_overclock_factor=None,
        override_reward_on_score=True,
        score_reward=0.0,
        goal_bonus=0.0,
        terminate_on_score=False,
        terminate_ball_below_z=0.1,
        fixed_object_position=None,
        floor_penalty=0.0,
    ):
        super().__init__(env)
        self.has_scored = False
        self._goal_bonus_awarded = False
        if throw_overclock_factor is None:
            throw_overclock_factor = DEFAULT_THROW_OVERCLOCK_FACTOR
        self.throw_overclock_factor = float(max(1.0, throw_overclock_factor))
        self.override_reward_on_score = bool(override_reward_on_score)
        self.score_reward = float(score_reward)
        self.goal_bonus = float(goal_bonus)
        self.terminate_on_score = bool(terminate_on_score)
        self.terminate_ball_below_z = (
            None if terminate_ball_below_z is None else float(terminate_ball_below_z)
        )
        self.fixed_object_position = (
            None
            if fixed_object_position is None
            else np.asarray(fixed_object_position, dtype=np.float64).copy()
        )
        # Positive value; subtracted from `reward` on the floor-termination step
        # unless the ball already scored. Counteracts the reward-hacking strategy
        # of dumping the ball off the table to escape -1/step early.
        self.floor_penalty = float(max(0.0, floor_penalty))
        _patch_fetch_pos_scale(self.env.unwrapped, self.throw_overclock_factor)

    def _force_object_position(self, position):
        """Overwrite the object's xyz in MuJoCo and refresh the env observation.

        The full (x, y, z) is written from `position`. Callers should pick a
        z that is slightly above the table surface so the ball falls under
        gravity and settles on the table during the policy's idle pre-grasp
        steps, rather than starting interpenetrated with the table mesh.

        Returns the refreshed obs dict on success, or None if the underlying
        env doesn't expose the modern (model, data) MuJoCo bindings — in which
        case the caller should fall back to the original obs.
        """
        fetch_env = self.env.unwrapped
        utils = getattr(fetch_env, "_utils", None)
        model = getattr(fetch_env, "model", None)
        data = getattr(fetch_env, "data", None)
        if utils is None or model is None or data is None:
            return None
        try:
            qpos = utils.get_joint_qpos(model, data, "object0:joint").copy()
            qpos[:3] = np.asarray(position, dtype=qpos.dtype)
            utils.set_joint_qpos(model, data, "object0:joint", qpos)
            # Zero out any residual object velocity from the previous step so
            # the ball starts at rest at the fixed position.
            qvel = utils.get_joint_qvel(model, data, "object0:joint")
            qvel[:] = 0.0
            utils.set_joint_qvel(model, data, "object0:joint", qvel)
            import mujoco
            mujoco.mj_forward(model, data)
            return fetch_env._get_obs()
        except Exception:
            return None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.has_scored = False
        self._goal_bonus_awarded = False

        if self.fixed_object_position is not None:
            forced_obs = self._force_object_position(self.fixed_object_position)
            if forced_obs is not None:
                obs = forced_obs

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
            if self.override_reward_on_score:
                reward = self.score_reward
            if (self.goal_bonus != 0.0) and (not self._goal_bonus_awarded):
                reward += self.goal_bonus
                self._goal_bonus_awarded = True
            if self.terminate_on_score:
                terminated = True

        if (
            self.terminate_ball_below_z is not None
            and float(ball_pos[2]) < self.terminate_ball_below_z
        ):
            terminated = True
            info["terminated_ball_floor"] = True
            if self.floor_penalty != 0.0 and not self.has_scored:
                reward -= self.floor_penalty
                info["floor_penalty_applied"] = float(self.floor_penalty)

        return obs, reward, terminated, truncated, info
