# free-throw-arm-bot
Simulated robot arm trained to shoot free throws through reinforcement learning (RL)

This project demonstrates the efficacy of imitation learning as an initialization step for RL. Specifically, this project tests the hypothesis that pre-training an RL policy with Behavioral Cloning (BC) on a small dataset of programmatic demonstrations will significantly reduce the exploration time (total timesteps to convergence) required to achieve a $>90\%$ task success rate, compared to an RL agent trained entirely from scratch.

## Notes

1. **Custom FetchPickAndPlace-v2 assets**  
   This project uses a custom modified FetchPickAndPlace-v2 environment (from the gymnasium-robotics package) to add the basketball hoop and backboard and to change the manipulated object to a ball (smaller than a regulation basketball relative to the hoop, which simplifies the task). The modified MJCF is in `pick_and_place.xml` at the repo root. To reproduce the environment, install the same version of gymnasium-robotics (1.2.2) in a venv, then replace `.venv\Lib\site-packages\gymnasium_robotics\envs\assets\fetch\pick_and_place.xml` with that file.

2. **Throw overclock factor (default 3.0)**  
   Gymnasium Fetch scales end-effector Cartesian deltas by a fixed `0.05` m per environment step inside `_set_action`, and clips actions to `[-1, 1]` before that. At full action magnitude, throws toward a distant hoop are often too weak for reliable baskets. The `FetchThrowWrapper` in `envs/fetch_throw_env.py` patches the unwrapped env so that internal scale becomes `0.05 × DEFAULT_THROW_OVERCLOCK_FACTOR` (currently **3.0**, larger values did not seem to be stable), which keeps the same 4D action interface and recorded demos but makes the simulated arm strong enough for this task. **Use this same wrapper (and thus the same factor) for expert data collection, BC training, and RL** so the MDP is consistent. You can override the factor by passing `throw_overclock_factor=` to `FetchThrowWrapper` if you need to experiment.
