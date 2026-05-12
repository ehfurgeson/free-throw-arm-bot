import gymnasium as gym
import numpy as np
import sys
import os

# Allow running as either `python -m scripts.train_rl_baseline` or script path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.expert_policy import compute_expert_action, reset_expert_state
from scripts.experiment_config import EnvironmentConfig
from envs.fetch_throw_env import FetchThrowWrapper

def collect_data(num_successes_needed=1000, max_episode_steps=200, throw_overclock_factor=None):
    base_env = gym.make('FetchPickAndPlace-v2', max_episode_steps=max_episode_steps)
    # None → DEFAULT_THROW_OVERCLOCK_FACTOR (3.0) in FetchThrowWrapper.
    # Pin the ball spawn to the same canonical position used by training/eval
    # so the BC dataset matches the env the policy will be deployed in.
    env = FetchThrowWrapper(
        base_env,
        throw_overclock_factor=throw_overclock_factor,
        fixed_object_position=EnvironmentConfig().fixed_object_position,
    )

    successful_trajectories = []
    successes_collected = 0
    attempts = 0

    while successes_collected < num_successes_needed:
        attempts += 1
        obs, info = env.reset()
        # The expert FSM relies on >0.15m ball position jumps to auto-detect
        # episode boundaries, which doesn't fire when the spawn is pinned.
        # Force a clean state at the top of every attempt so each episode is
        # independent of the previous one.
        reset_expert_state()
        # [DIAGNOSTIC] Confirm the fixed-spawn override is taking effect and
        # show ball position right after reset. Remove once verified.
        print(
            f"attempt={attempts} reset ball={np.round(obs['observation'][3:6], 4)} "
            f"gripper={np.round(obs['observation'][0:3], 4)}",
            flush=True,
        )
        done = False
        truncated = False
        step_idx = 0

        current_trajectory = {'obs': [], 'actions': []}

        while not (done or truncated):
            # 1. Ask the expert from Step 1.3 what to do
            action = compute_expert_action(obs)

            # 2. Save the state and the action
            current_trajectory['obs'].append(obs['observation'])
            current_trajectory['actions'].append(action)

            # 3. Take the step in the environment
            obs, reward, done, truncated, info = env.step(action)
            step_idx += 1

        # [DIAGNOSTIC] Surface why the episode ended so we can tell apart "ball
        # squirted off the table", "expert stuck in approach", and "scored".
        # Remove once verified.
        last_ball_z = float(obs['observation'][5])
        scored = info.get('is_success', 0.0) == 1.0
        end_reason = (
            "scored" if scored
            else ("ball_floor" if info.get("terminated_ball_floor") else
                  ("truncated" if truncated else "terminated_other"))
        )
        # [DIAGNOSTIC] Peek at the expert FSM state to detect "stuck in
        # follow_through across episodes" pathology.
        expert_state = getattr(compute_expert_action, "_state", "<uninit>")
        open_sign = getattr(compute_expert_action, "_open_sign", "<uninit>")
        # [DIAGNOSTIC] Spatial relationship at end of episode: where is the
        # gripper, where is the ball, and how far apart are they in xy / z?
        end_gripper = np.round(obs['observation'][0:3], 4)
        end_ball = np.round(obs['observation'][3:6], 4)
        end_xy = float(np.linalg.norm(obs['observation'][6:8]))   # object_rel_pos[xy]
        end_dz = float(obs['observation'][8])                     # object_rel_pos[z]
        print(
            f"  -> end steps={step_idx} reason={end_reason} fsm={expert_state} "
            f"open_sign={open_sign}",
            flush=True,
        )
        print(
            f"     gripper={end_gripper} ball={end_ball} "
            f"xy_err={end_xy:.4f} dz(ball-gripper)={end_dz:.4f}",
            flush=True,
        )

        # 4. Check if the throw was actually a success
        if scored:
            successful_trajectories.append(current_trajectory)
            successes_collected += 1
            print(f"Success! Collected {successes_collected}/{num_successes_needed}", flush=True)
            
    # 5. Save out to a file format your BC script can read
    np.savez('data/expert_demonstrations.npz', trajectories=successful_trajectories)

if __name__ == "__main__":
    collect_data()