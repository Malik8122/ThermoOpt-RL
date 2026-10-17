"""Deep Q-Network (Step 10), written by hand in PyTorch.

Input is the 30-feature normalised observation of the *feature* environment (never the 144-state id). Output is
18 Q-values, one per primitive action (placement x cooling).

    online, target:  Linear(30,128) - ReLU - Linear(128,128) - ReLU - Linear(128,18); target <- online at start
    act:             epsilon-greedy over the live mask M(s) = env.action_masks()
                     epsilon(t) = max(eps_end, eps_start - (eps_start - eps_end) * t / eps_decay_steps),
                     t = environment steps so far (linear, per step)
    store:           (s, a, r, s', terminated, truncated, M(s), M(s')) in the replay buffer
    learn:           after `learning_starts` stored transitions, every `train_frequency` steps:
                     y = r + gamma * (1 - d) * max_{a' in M(s')} Q_target(s', a')
                     d = terminated OR truncated   (config: truncation_is_terminal)
                     loss = Huber(Q_online(s, a), y); Adam; gradient-norm clip
    target sync:     hard copy every `target_update_frequency` gradient updates

Masking ablation (``use_action_mask = False``): exploration, the greedy choice and the target's max all range
over the 18 actions. The environment still applies its own invalid-action semantics (a no-op plus a penalty).

    python -m backend.rl.dqn                 # full protocol -> models/dqn/, experiments/dqn_training.csv
    python -m backend.rl.dqn --no-mask       # ablation      -> models/dqn_nomask/, experiments/dqn_nomask_training.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from backend.config import Config, DQNConfig, get_config
from backend.evaluation.runner import episode_metrics
from backend.rl.baselines import Controller
from backend.rl.environment import DataCenterEnv
from backend.rl.replay_buffer import ReplayBuffer
from backend.utils.logging_utils import get_logger
from backend.utils.seeding import derive_seed, make_rng, set_global_seed

logger = get_logger("dqn")

CHECKPOINT = "dqn.pt"
META_FILE = "agent.json"


def configure_torch(threads: int) -> None:
    """Deterministic CPU execution: fixed thread count, deterministic kernels."""
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "cuda":
        logger.warning("CUDA requested but not available; using CPU")
    return torch.device("cpu")


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes: list[int]) -> None:
        super().__init__()
        layers, width = [], obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(width, h), nn.ReLU()]
            width = h
        layers.append(nn.Linear(width, n_actions))
        self.net = nn.Sequential(*layers)
        self.obs_dim, self.n_actions = obs_dim, n_actions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.obs_dim:
            raise ValueError(f"expected {self.obs_dim} input features, got {x.shape[-1]}")
        return self.net(x)


class DQNAgent:
    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes: list[int], learning_rate: float, gamma: float,
                 batch_size: int, replay_buffer_size: int, target_update_frequency: int, epsilon_start: float,
                 epsilon_end: float, epsilon_decay_steps: int, learning_starts: int, train_frequency: int,
                 max_grad_norm: float, loss: str = "huber", use_action_mask: bool = True,
                 truncation_is_terminal: bool = True, seed: int = 42, device: str = "cpu") -> None:
        self.obs_dim, self.n_actions, self.hidden_sizes = obs_dim, n_actions, list(hidden_sizes)
        self.learning_rate, self.gamma, self.batch_size = learning_rate, gamma, batch_size
        self.replay_buffer_size, self.target_update_frequency = replay_buffer_size, target_update_frequency
        self.epsilon_start, self.epsilon_end, self.epsilon_decay_steps = epsilon_start, epsilon_end, epsilon_decay_steps
        self.learning_starts = max(learning_starts, batch_size)
        self.train_frequency, self.max_grad_norm, self.loss_name = train_frequency, max_grad_norm, loss
        self.use_action_mask, self.truncation_is_terminal = use_action_mask, truncation_is_terminal
        self.seed, self.device = seed, resolve_device(device)

        torch.manual_seed(derive_seed(seed, "dqn", "init"))
        self.online = QNetwork(obs_dim, n_actions, hidden_sizes).to(self.device)
        self.target = QNetwork(obs_dim, n_actions, hidden_sizes).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss() if loss == "huber" else nn.MSELoss()
        self.buffer = ReplayBuffer(replay_buffer_size, obs_dim, n_actions)
        self.explore_rng = make_rng(derive_seed(seed, "dqn", "exploration"))
        self.replay_rng = make_rng(derive_seed(seed, "dqn", "replay"))
        self.env_steps = self.training_updates = self.target_updates = self.episodes_trained = 0

    @classmethod
    def from_config(cls, cfg: DQNConfig, obs_dim: int = 30, n_actions: int = 18, **overrides) -> "DQNAgent":
        params = dict(hidden_sizes=cfg.hidden_sizes, learning_rate=cfg.learning_rate, gamma=cfg.gamma,
                      batch_size=cfg.batch_size, replay_buffer_size=cfg.replay_buffer_size,
                      target_update_frequency=cfg.target_update_frequency, epsilon_start=cfg.epsilon_start,
                      epsilon_end=cfg.epsilon_end, epsilon_decay_steps=cfg.epsilon_decay_steps,
                      learning_starts=cfg.learning_starts, train_frequency=cfg.train_frequency,
                      max_grad_norm=cfg.max_grad_norm, loss=cfg.loss, use_action_mask=cfg.use_action_mask,
                      truncation_is_terminal=cfg.truncation_is_terminal, seed=cfg.seed, device=cfg.device)
        params.update(overrides)
        return cls(obs_dim, n_actions, **params)

    # ------------------------------------------------------------ acting
    def epsilon(self, step: int | None = None) -> float:
        t = self.env_steps if step is None else step
        frac = min(1.0, t / self.epsilon_decay_steps)
        return max(self.epsilon_end, self.epsilon_start - (self.epsilon_start - self.epsilon_end) * frac)

    def _check_obs(self, obs) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.shape != (self.obs_dim,):
            raise ValueError(f"observation must have shape ({self.obs_dim},), got {obs.shape}")
        return obs

    def q_values(self, obs) -> np.ndarray:
        with torch.no_grad():
            x = torch.as_tensor(self._check_obs(obs), device=self.device).unsqueeze(0)
            return self.online(x).squeeze(0).cpu().numpy().astype(np.float64)

    def _valid(self, mask) -> np.ndarray:
        if mask is None or not self.use_action_mask:
            return np.ones(self.n_actions, dtype=bool)
        mask = np.asarray(mask, dtype=bool)
        if not mask.any():
            raise ValueError("action mask has no valid action")
        return mask

    def greedy_action(self, obs, mask=None) -> int:
        """argmax of Q over the valid actions; exact ties -> lowest index."""
        q = np.where(self._valid(mask), self.q_values(obs), -np.inf)
        return int(np.argmax(q))

    def select_action(self, obs, mask, epsilon: float) -> tuple[int, bool]:
        valid = self._valid(mask)
        if epsilon > 0 and self.explore_rng.random() < epsilon:
            return int(self.explore_rng.choice(np.flatnonzero(valid))), True
        return self.greedy_action(obs, valid), False

    def predict(self, obs, mask=None) -> int:
        return self.greedy_action(obs, mask)

    # ----------------------------------------------------------- learning
    def store_transition(self, state, action, reward, next_state, terminated, truncated, mask, next_mask) -> None:
        self.buffer.add(state, action, reward, next_state, terminated, truncated, mask, next_mask)

    def compute_targets(self, rewards: torch.Tensor, next_states: torch.Tensor, terminated: torch.Tensor,
                        truncated: torch.Tensor, next_masks: torch.Tensor) -> torch.Tensor:
        """y = r + gamma * (1 - d) * max over valid a' of Q_target(s', a')."""
        with torch.no_grad():
            next_q = self.target(next_states)
            if self.use_action_mask:
                next_q = next_q.masked_fill(~next_masks, float("-inf"))
            next_max = next_q.max(dim=1).values
            done = terminated | truncated if self.truncation_is_terminal else terminated
            return rewards + self.gamma * torch.where(done, torch.zeros_like(next_max), next_max)

    def train_step(self) -> tuple[float, float]:
        """One gradient update on a sampled batch. Returns (loss, mean predicted Q of the taken actions)."""
        b = self.buffer.sample(self.batch_size, self.replay_rng)
        t = lambda x, dtype=None: torch.as_tensor(x, device=self.device, dtype=dtype)  # noqa: E731
        states, actions = t(b.states), t(b.actions, torch.int64)
        y = self.compute_targets(t(b.rewards), t(b.next_states), t(b.terminated), t(b.truncated), t(b.next_masks))
        q_pred = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = self.loss_fn(q_pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.training_updates += 1
        if self.training_updates % self.target_update_frequency == 0:
            self.update_target_network()
        loss_value, q_mean = float(loss.item()), float(q_pred.mean().item())
        if not (np.isfinite(loss_value) and np.isfinite(q_mean)):
            raise FloatingPointError(f"non-finite loss/Q at update {self.training_updates}")
        return loss_value, q_mean

    def update_target_network(self) -> None:
        self.target.load_state_dict(self.online.state_dict())
        self.target_updates += 1

    def learn_episode(self, env, env_seed: int | None = None) -> dict:
        obs, info = env.reset(seed=env_seed)
        obs = self._check_obs(obs)
        space = getattr(env, "observation_space", None)
        if space is not None and hasattr(space, "contains") and not space.contains(obs):
            raise ValueError("observation outside the environment's observation space")
        mask = np.asarray(info["action_mask"] if "action_mask" in info else env.action_masks(), dtype=bool)
        losses, qs = [], []
        steps = explored = invalid = 0
        temperature_sum = 0.0
        done = False
        while not done:
            action, was_explore = self.select_action(obs, mask, self.epsilon())
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_obs = self._check_obs(next_obs)
            next_mask = np.asarray(env.action_masks(), dtype=bool)
            self.store_transition(obs, action, float(reward), next_obs, terminated, truncated, mask, next_mask)
            self.env_steps += 1
            steps += 1
            explored += was_explore
            invalid += not info.get("action_valid", True)
            temperature_sum += info.get("avg_temperature", 0.0)
            if len(self.buffer) >= self.learning_starts and self.env_steps % self.train_frequency == 0:
                loss, q = self.train_step()
                losses.append(loss)
                qs.append(q)
            obs, mask = next_obs, next_mask
            done = bool(terminated or truncated)
        self.episodes_trained += 1
        return {"steps": steps, "epsilon": self.epsilon(), "explore_steps": explored,
                "exploit_steps": steps - explored, "invalid_actions": invalid,
                "invalid_action_rate": invalid / steps,
                "mean_loss": float(np.mean(losses)) if losses else float("nan"),
                "max_loss": float(np.max(losses)) if losses else float("nan"),
                "mean_q": float(np.mean(qs)) if qs else float("nan"),
                "replay_size": len(self.buffer), "training_updates": self.training_updates,
                "target_updates": self.target_updates, "env_steps": self.env_steps,
                "_last_info": info, "_temperature_sum": temperature_sum}

    # -------------------------------------------------------- persistence
    def hyperparameters(self) -> dict:
        return {"hidden_sizes": self.hidden_sizes, "learning_rate": self.learning_rate, "gamma": self.gamma,
                "batch_size": self.batch_size, "replay_buffer_size": self.replay_buffer_size,
                "target_update_frequency": self.target_update_frequency, "epsilon_start": self.epsilon_start,
                "epsilon_end": self.epsilon_end, "epsilon_decay_steps": self.epsilon_decay_steps,
                "learning_starts": self.learning_starts, "train_frequency": self.train_frequency,
                "max_grad_norm": self.max_grad_norm, "loss": self.loss_name, "use_action_mask": self.use_action_mask,
                "truncation_is_terminal": self.truncation_is_terminal, "seed": self.seed}

    def save(self, directory: Path, metadata: dict | None = None) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        counters = {"env_steps": self.env_steps, "training_updates": self.training_updates,
                    "target_updates": self.target_updates, "episodes_trained": self.episodes_trained}
        torch.save({"online": self.online.state_dict(), "target": self.target.state_dict(),
                    "optimizer": self.optimizer.state_dict(), "counters": counters,
                    "obs_dim": self.obs_dim, "n_actions": self.n_actions,
                    "hyperparameters": self.hyperparameters()}, directory / CHECKPOINT)
        payload = {"algorithm": "dqn", "obs_dim": self.obs_dim, "n_actions": self.n_actions,
                   "architecture": repr(self.online.net), "device": str(self.device),
                   "epsilon_at_save": self.epsilon(), **counters, "hyperparameters": self.hyperparameters(),
                   "checkpoint": CHECKPOINT, **(metadata or {})}
        (directory / META_FILE).write_text(json.dumps(payload, indent=1), encoding="utf-8")
        return directory / META_FILE

    @classmethod
    def load(cls, directory: Path, device: str = "cpu") -> "DQNAgent":
        directory = Path(directory)
        ckpt = torch.load(directory / CHECKPOINT, map_location="cpu", weights_only=True)
        agent = cls(ckpt["obs_dim"], ckpt["n_actions"], device=device, **ckpt["hyperparameters"])
        agent.online.load_state_dict(ckpt["online"])
        agent.target.load_state_dict(ckpt["target"])
        agent.optimizer.load_state_dict(ckpt["optimizer"])
        for k, v in ckpt["counters"].items():
            setattr(agent, k, v)
        agent.metadata = json.loads((directory / META_FILE).read_text(encoding="utf-8"))
        return agent


class DQNController(Controller):
    """Greedy (epsilon = 0) DQN for the shared runner. Uses the live mask unless the agent is the no-mask ablation.
    Optionally records observations, masks and actions for analysis."""

    name = "dqn"

    def __init__(self, agent: DQNAgent, config: Config | None = None, record: bool = False) -> None:
        super().__init__(config)
        self.agent, self.epsilon, self.record = agent, 0.0, record
        self.action_counts = np.zeros(agent.n_actions, dtype=np.int64)
        self.observations: list[np.ndarray] = []
        self.masks: list[np.ndarray] = []
        self.actions: list[int] = []

    def act(self, observation, action_mask=None) -> int:
        action, explored = self.agent.select_action(observation, action_mask, self.epsilon)
        assert not explored
        self.action_counts[action] += 1
        if self.record:
            self.observations.append(np.asarray(observation, dtype=np.float32))
            self.masks.append(np.asarray(action_mask, dtype=bool))
            self.actions.append(action)
        return action


# --------------------------------------------------------------------------- #
# Data-centre training
# --------------------------------------------------------------------------- #
CSV_FIELDS = ("episode", "env_seed", "start_hour", "epsilon", "total_reward", "energy_kwh", "it_energy_kwh",
              "cooling_energy_kwh", "carbon_kg", "cost_inr", "sla_failures", "completed_jobs", "critical_rack_minutes",
              "warning_or_worse_rack_minutes", "peak_temperature_c", "invalid_actions", "invalid_action_rate",
              "explore_steps", "exploit_steps", "mean_loss", "max_loss", "mean_q", "replay_size",
              "training_updates", "target_updates", "env_steps")


def train_data_centre(agent: DQNAgent, config: Config, episodes: int, episode_minutes: int,
                      randomize_start_hour: bool, log_every: int = 10) -> list[dict]:
    env = DataCenterEnv(representation="features", episode_minutes=episode_minutes,
                        randomize_start_hour=randomize_start_hour, config=config)
    assert env.observation_space.shape == (agent.obs_dim,) and env.action_space.n == agent.n_actions
    rows, start = [], time.perf_counter()
    for e in range(episodes):
        env_seed = derive_seed(agent.seed, "dqn", "episode", e)
        stats = agent.learn_episode(env, env_seed)
        metrics = episode_metrics(env, stats.pop("_last_info"), stats.pop("_temperature_sum"), env_seed)
        row = {"episode": e, "env_seed": env_seed, "start_hour": env.dc.start_hour, **metrics, **stats}
        rows.append({k: row[k] for k in CSV_FIELDS})
        if log_every and ((e + 1) % log_every == 0 or e == episodes - 1):
            recent = rows[-log_every:]
            logger.info("%s episode %d/%d  eps %.3f  reward %.1f (last %d mean %.1f)  SLA %d  crit %d  loss %.4f  "
                        "Q %.2f  invalid %.3f  %.0f s", "mask" if agent.use_action_mask else "no-mask", e + 1,
                        episodes, row["epsilon"], row["total_reward"], len(recent),
                        np.mean([r["total_reward"] for r in recent]), row["sla_failures"],
                        row["critical_rack_minutes"], row["mean_loss"], row["mean_q"], row["invalid_action_rate"],
                        time.perf_counter() - start)
    return rows


def write_training_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    cfg = get_config()
    d = cfg.rl.dqn
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-mask", action="store_true", help="masking ablation (use_action_mask = False)")
    parser.add_argument("--episodes", type=int, default=d.episodes)
    parser.add_argument("--minutes", type=int, default=d.episode_minutes)
    parser.add_argument("--seed", type=int, default=d.seed)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()
    tag = "dqn_nomask" if args.no_mask else "dqn"
    out = args.out or cfg.paths.models / tag
    csv_path = args.csv or cfg.paths.experiments / f"{tag}_training.csv"
    full = args.episodes == d.episodes and args.minutes == d.episode_minutes
    if not full:
        logger.warning("development run: %d x %d min differs from the configured protocol %d x %d min",
                       args.episodes, args.minutes, d.episodes, d.episode_minutes)

    set_global_seed(args.seed)
    configure_torch(d.torch_threads)
    agent = DQNAgent.from_config(d, seed=args.seed, use_action_mask=not args.no_mask)
    started = time.perf_counter()
    rows = train_data_centre(agent, cfg, args.episodes, args.minutes, d.randomize_start_hour)
    seconds = time.perf_counter() - started
    write_training_csv(rows, csv_path)
    agent.save(out, metadata={"training": {
        "episodes": args.episodes, "episode_minutes": args.minutes, "total_env_steps": agent.env_steps,
        "training_updates": agent.training_updates, "target_updates": agent.target_updates,
        "full_protocol": full, "seconds": seconds, "device": str(agent.device), "torch_threads": d.torch_threads,
        "torch_version": torch.__version__, "randomize_start_hour": d.randomize_start_hour,
        "env_seed_strategy": "episode e: derive_seed(seed, 'dqn', 'episode', e)",
        "rng_streams": "init: derive_seed(seed,'dqn','init'); exploration: ...'exploration'; replay: ...'replay'",
        "training_csv": str(csv_path)}})
    print(f"DQN ({'no mask' if args.no_mask else 'masked'}): {args.episodes} x {args.minutes} min = {agent.env_steps} "
          f"steps, {agent.training_updates} updates, {agent.target_updates} target syncs, {seconds:.0f} s on "
          f"{agent.device}; final epsilon {agent.epsilon():.3f}")
    logger.info("wrote %s and %s", out, csv_path)


if __name__ == "__main__":
    main()
