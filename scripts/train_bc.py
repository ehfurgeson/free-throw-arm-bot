import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

# Allow running as either `python -m scripts.train_bc` or script path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.experiment_config import BCConfig, make_dense_only_env_config
from scripts.training_utils import (
    ensure_dir,
    evaluate_policy_callable,
    load_expert_transitions,
    reseed,
    save_curve_plot,
    select_device,
    write_json,
)


class BCPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_size = int(hidden_size)
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


class _BCEvalPolicy:
    """Stateful adapter: feeds a frame-stacked window of env observations and
    the previous action to a BCPolicy.

    `reset()` is invoked by `evaluate_policy_callable` between episodes so the
    frame buffer and prev-action memory clear at episode boundaries.
    """

    def __init__(
        self,
        model: BCPolicy,
        device: torch.device,
        action_dim: int,
        frame_stack: int = 1,
    ):
        self.model = model
        self.device = device
        self.action_dim = int(action_dim)
        self.frame_stack = int(max(1, frame_stack))
        self.env_obs_dim: int | None = None
        self.prev_action = np.zeros(self.action_dim, dtype = np.float32)
        self.obs_history: list[np.ndarray] = []

    def reset(self) -> None:
        self.prev_action = np.zeros(self.action_dim, dtype = np.float32)
        self.obs_history = []

    def _stacked_obs(self, env_obs: np.ndarray) -> np.ndarray:
        self.obs_history.append(env_obs)
        if len(self.obs_history) > self.frame_stack:
            self.obs_history = self.obs_history[-self.frame_stack :]
        pad_count = self.frame_stack - len(self.obs_history)
        if pad_count > 0:
            pad = [np.zeros(env_obs.shape[0], dtype = np.float32)] * pad_count
            return np.concatenate(pad + self.obs_history, axis = 0)
        return np.concatenate(self.obs_history, axis = 0)

    def __call__(self, obs_dict) -> np.ndarray:
        env_obs = np.asarray(obs_dict["observation"], dtype = np.float32)
        if self.env_obs_dim is None:
            self.env_obs_dim = int(env_obs.shape[0])
        stacked = self._stacked_obs(env_obs)
        x = np.concatenate([stacked, self.prev_action], axis = 0)
        with torch.no_grad():
            x_t = torch.as_tensor(x, dtype = torch.float32, device = self.device).unsqueeze(0)
            action = self.model(x_t).squeeze(0).detach().cpu().numpy()
        self.prev_action = action.astype(np.float32)
        return action


def evaluate_bc_model(
    model: BCPolicy,
    device: torch.device,
    eval_episodes: int,
    seed: int,
    frame_stack: int = 1,
) -> dict:
    model.eval()

    policy = _BCEvalPolicy(
        model = model,
        device = device,
        action_dim = model.action_dim,
        frame_stack = frame_stack,
    )

    return evaluate_policy_callable(
        policy_fn = policy,
        env_cfg = make_dense_only_env_config(),
        seed = seed,
        n_episodes = eval_episodes,
    )


def train_bc(npz_path: str, output_root: str, seed: int, eval_episodes: int, bc_cfg: BCConfig):
    reseed(seed)
    device = select_device()
    out_dir = ensure_dir(Path(output_root) / "bc")

    print(f"[bc] device={device} dataset={npz_path} output={out_dir}", flush=True)

    obs, actions = load_expert_transitions(
        npz_path,
        include_prev_action = True,
        frame_stack = bc_cfg.frame_stack,
    )
    assert obs.shape[0] > 0, "No transitions found in expert dataset."
    assert actions.shape[0] > 0, "No actions found in expert dataset."
    assert obs.shape[0] ==  actions.shape[0], "Mismatch between observation and action counts."

    print(
        f"[bc] loaded transitions={obs.shape[0]} obs_dim={obs.shape[1]} "
        f"action_dim={actions.shape[1]} frame_stack={bc_cfg.frame_stack}",
        flush=True,
    )

    obs_t = torch.as_tensor(obs, dtype = torch.float32)
    act_t = torch.as_tensor(actions, dtype = torch.float32)
    dataset = TensorDataset(obs_t, act_t)

    val_size = int(len(dataset) * bc_cfg.val_split)
    val_size = max(1, val_size)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator = torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(train_ds, batch_size = bc_cfg.batch_size, shuffle = True)
    val_loader = DataLoader(val_ds, batch_size = bc_cfg.batch_size, shuffle = False)

    model = BCPolicy(obs_dim = obs.shape[1], action_dim = actions.shape[1], hidden_size = bc_cfg.hidden_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr = bc_cfg.learning_rate)
    criterion = nn.MSELoss()

    # Write the config eagerly so mid-training viz can pick up the correct
    # architecture (e.g. augmented obs_dim, frame_stack) from disk.
    bc_config_payload = {
        "seed": seed,
        "batch_size": bc_cfg.batch_size,
        "learning_rate": bc_cfg.learning_rate,
        "train_steps": bc_cfg.train_steps,
        "val_split": bc_cfg.val_split,
        "early_stop_patience": bc_cfg.early_stop_patience,
        "hidden_size": bc_cfg.hidden_size,
        "obs_dim": int(obs.shape[1]),
        "action_dim": int(actions.shape[1]),
        "include_prev_action": True,
        "frame_stack": int(bc_cfg.frame_stack),
    }
    write_json(out_dir / "bc_config.json", bc_config_payload)

    history = {
        "step": [],
        "train_loss": [],
        "val_loss": [],
        "eval_success_rate": [],
        "eval_mean_return": [],
    }
    best_val = float("inf")
    patience_counter = 0

    print(
        f"[bc] starting training: train_steps={bc_cfg.train_steps} "
        f"batch_size={bc_cfg.batch_size} lr={bc_cfg.learning_rate} "
        f"val_split={bc_cfg.val_split} eval_every_steps={bc_cfg.eval_every_steps} "
        f"hidden_size={bc_cfg.hidden_size}",
        flush=True,
    )

    global_step = 0
    epoch_idx = 0
    while global_step < bc_cfg.train_steps:
        model.train()
        train_losses = []
        for batch_obs, batch_actions in train_loader:
            batch_obs = batch_obs.to(device)
            batch_actions = batch_actions.to(device)
            pred_actions = model(batch_obs)
            loss = criterion(pred_actions, batch_actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            global_step +=  1
            if global_step >=  bc_cfg.train_steps:
                break

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_obs, batch_actions in val_loader:
                batch_obs = batch_obs.to(device)
                batch_actions = batch_actions.to(device)
                pred_actions = model(batch_obs)
                val_loss = criterion(pred_actions, batch_actions)
                val_losses.append(val_loss.item())

        mean_train_loss = float(np.mean(train_losses))
        mean_val_loss = float(np.mean(val_losses))
        history["step"].append(global_step)
        history["train_loss"].append(mean_train_loss)
        history["val_loss"].append(mean_val_loss)
        epoch_idx += 1

        ran_eval = False
        eval_success_rate = None
        eval_mean_return = None
        if global_step % bc_cfg.eval_every_steps ==  0 or global_step ==  bc_cfg.train_steps:
            eval_metrics = evaluate_bc_model(
                model = model,
                device = device,
                eval_episodes = eval_episodes,
                seed = seed + 999,
                frame_stack = bc_cfg.frame_stack,
            )
            history["eval_success_rate"].append(eval_metrics["success_rate"])
            history["eval_mean_return"].append(eval_metrics["mean_return"])
            write_json(out_dir / "latest_eval.json", eval_metrics)
            ran_eval = True
            eval_success_rate = eval_metrics["success_rate"]
            eval_mean_return = eval_metrics["mean_return"]

        improved = mean_val_loss < best_val
        if improved:
            best_val = mean_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), out_dir / "best_bc_policy.pt")
        else:
            patience_counter +=  1

        # Per-epoch one-liner so the user can confirm the run is actually
        # making progress (steps/total, losses, optional eval, best-val flag,
        # early-stop patience). Kept on a single line for easy log scraping.
        progress_pct = (global_step / max(1, bc_cfg.train_steps)) * 100.0
        msg = (
            f"[bc] epoch={epoch_idx:3d} step={global_step}/{bc_cfg.train_steps} "
            f"({progress_pct:5.1f}%) train_loss={mean_train_loss:.5f} "
            f"val_loss={mean_val_loss:.5f} best_val={best_val:.5f}"
        )
        if ran_eval:
            msg += f" eval_success={eval_success_rate:.2f} eval_return={eval_mean_return:.2f}"
        if improved:
            msg += " [best ↓]"
        else:
            msg += f" patience={patience_counter}/{bc_cfg.early_stop_patience}"
        print(msg, flush=True)

        if not improved and patience_counter >=  bc_cfg.early_stop_patience:
            print(
                f"[bc] early stop at step={global_step} "
                f"(val hasn't improved for {patience_counter} epochs)",
                flush=True,
            )
            break

    torch.save(model.state_dict(), out_dir / "final_bc_policy.pt")
    write_json(out_dir / "bc_history.json", history)
    write_json(out_dir / "bc_config.json", bc_config_payload)

    final_success = history["eval_success_rate"][-1] if history["eval_success_rate"] else None
    print(
        f"[bc] training complete: steps={global_step} epochs={epoch_idx} "
        f"best_val={best_val:.5f}"
        + (f" final_eval_success={final_success:.2f}" if final_success is not None else "")
        + f" saved to {out_dir}",
        flush=True,
    )

    save_curve_plot(
        x = history["step"],
        y = history["train_loss"],
        title = "BC Train Loss",
        xlabel = "steps",
        ylabel = "mse_loss",
        out_path = out_dir / "bc_train_loss.png",
    )
    save_curve_plot(
        x = history["step"],
        y = history["val_loss"],
        title = "BC Validation Loss",
        xlabel = "steps",
        ylabel = "mse_loss",
        out_path = out_dir / "bc_val_loss.png",
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type = str, default = "data/expert_demonstrations.npz")
    parser.add_argument("--output-root", type = str, default = "artifacts")
    parser.add_argument("--seed", type = int, default = 7)
    parser.add_argument("--eval-episodes", type = int, default = 20)
    parser.add_argument("--train-steps", type = int, default = 30_000)
    parser.add_argument("--batch-size", type = int, default = 256)
    parser.add_argument("--learning-rate", type = float, default = 3e-4)
    parser.add_argument("--val-split", type = float, default = 0.2)
    parser.add_argument("--early-stop-patience", type = int, default = 30)
    parser.add_argument("--eval-every-steps", type = int, default = 200)
    parser.add_argument("--hidden-size", type = int, default = 256)
    parser.add_argument(
        "--frame-stack",
        type = int,
        default = 4,
        help = "Number of consecutive env observations stacked as policy input (>=1).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    bc_cfg = BCConfig(
        batch_size = args.batch_size,
        learning_rate = args.learning_rate,
        train_steps = args.train_steps,
        val_split = args.val_split,
        early_stop_patience = args.early_stop_patience,
        eval_every_steps = args.eval_every_steps,
        hidden_size = args.hidden_size,
        frame_stack = max(1, args.frame_stack),
    )
    train_bc(
        npz_path = args.dataset_path,
        output_root = args.output_root,
        seed = args.seed,
        eval_episodes = args.eval_episodes,
        bc_cfg = bc_cfg,
    )


if __name__ ==  "__main__":
    main()
