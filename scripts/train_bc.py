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


def evaluate_bc_model(model: BCPolicy, device: torch.device, eval_episodes: int, seed: int) -> dict:
    model.eval()

    def _policy(obs_dict):
        obs = np.asarray(obs_dict["observation"], dtype = np.float32)
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype = torch.float32, device = device).unsqueeze(0)
            action = model(obs_t).squeeze(0).detach().cpu().numpy()
        return action

    return evaluate_policy_callable(
        policy_fn = _policy,
        env_cfg = make_dense_only_env_config(),
        seed = seed,
        n_episodes = eval_episodes,
    )


def train_bc(npz_path: str, output_root: str, seed: int, eval_episodes: int, bc_cfg: BCConfig):
    reseed(seed)
    device = select_device()
    out_dir = ensure_dir(Path(output_root) / "bc")

    obs, actions = load_expert_transitions(npz_path)
    assert obs.shape[0] > 0, "No transitions found in expert dataset."
    assert actions.shape[0] > 0, "No actions found in expert dataset."
    assert obs.shape[0] ==  actions.shape[0], "Mismatch between observation and action counts."

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

    history = {
        "step": [],
        "train_loss": [],
        "val_loss": [],
        "eval_success_rate": [],
        "eval_mean_return": [],
    }
    best_val = float("inf")
    patience_counter = 0

    global_step = 0
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

        if global_step % bc_cfg.eval_every_steps ==  0 or global_step ==  bc_cfg.train_steps:
            eval_metrics = evaluate_bc_model(
                model = model,
                device = device,
                eval_episodes = eval_episodes,
                seed = seed + 999,
            )
            history["eval_success_rate"].append(eval_metrics["success_rate"])
            history["eval_mean_return"].append(eval_metrics["mean_return"])
            write_json(out_dir / "latest_eval.json", eval_metrics)

        if mean_val_loss < best_val:
            best_val = mean_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), out_dir / "best_bc_policy.pt")
        else:
            patience_counter +=  1
            if patience_counter >=  bc_cfg.early_stop_patience:
                break

    torch.save(model.state_dict(), out_dir / "final_bc_policy.pt")
    write_json(out_dir / "bc_history.json", history)
    write_json(
        out_dir / "bc_config.json",
        {
            "seed": seed,
            "batch_size": bc_cfg.batch_size,
            "learning_rate": bc_cfg.learning_rate,
            "train_steps": bc_cfg.train_steps,
            "val_split": bc_cfg.val_split,
            "early_stop_patience": bc_cfg.early_stop_patience,
            "hidden_size": bc_cfg.hidden_size,
        },
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
