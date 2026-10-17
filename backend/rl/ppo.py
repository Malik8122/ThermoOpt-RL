"""Proximal Policy Optimisation (Step 11), written by hand in PyTorch.

Input is the 30-feature observation; the policy is a categorical distribution over the 18 primitive actions.

    actor   Linear(30,128) - Tanh - Linear(128,128) - Tanh - Linear(128,18)   -> logits
    critic  Linear(30,128) - Tanh - Linear(128,128) - Tanh - Linear(128,1)    -> V(s)
            (separate networks, orthogonal initialisation)

    policy      pi(a|s) = softmax(logits) restricted to the live mask M(s): invalid actions get probability 0
    rollout     collect `rollout_steps` transitions with the current policy (episodes continue across rollouts)
    advantages  GAE:  delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
                      A_t     = delta_t + gamma * lambda * A_{t+1}        (reset at episode ends)
                V(s_{t+1}) = 0 after a terminal step. After a time-limit truncation it is V(final observation)
                (config: truncation_is_terminal = false), so the 24 h limit is not mistaken for an ending.
    update      `update_epochs` passes over the rollout in shuffled minibatches:
                ratio = pi_new(a|s) / pi_old(a|s)
                L = -min(ratio * A, clip(ratio, 1-eps, 1+eps) * A) + c_v * 0.5 * (R - V)^2 - c_e * H[pi]
                advantages normalised per minibatch; Adam; gradient-norm clip

Masking ablation (``use_action_mask = False``): the policy spreads probability over all 18 actions and the
environment applies its own invalid-action semantics (a no-op plus a penalty).

    python -m backend.rl.ppo                                   # design protocol -> models/ppo/
    python -m backend.rl.ppo --no-mask                         # ablation        -> models/ppo_nomask/
    python -m backend.rl.ppo --timesteps 432000 --tag ppo_432k # DQN-matched budget -> models/ppo_432k/
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

from backend.config import Config, PPOConfig, get_config
from backend.evaluation.runner import episode_metrics
from backend.rl.baselines import Controller
from backend.rl.dqn import configure_torch, resolve_device
from backend.rl.environment import DataCenterEnv
from backend.utils.logging_utils import get_logger
from backend.utils.seeding import derive_seed, make_rng, set_global_seed

logger = get_logger("ppo")

CHECKPOINT = "ppo.pt"
META_FILE = "agent.json"


def _mlp(in_dim: int, hidden: list[int], out_dim: int, out_gain: float) -> nn.Sequential:
    layers, width = [], in_dim
    for h in hidden:
        lin = nn.Linear(width, h)
        nn.init.orthogonal_(lin.weight, gain=np.sqrt(2.0))
        nn.init.zeros_(lin.bias)
        layers += [lin, nn.Tanh()]
        width = h
    head = nn.Linear(width, out_dim)
    nn.init.orthogonal_(head.weight, gain=out_gain)
    nn.init.zeros_(head.bias)
    return nn.Sequential(*layers, head)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes: list[int]) -> None:
        super().__init__()
        self.obs_dim, self.n_actions = obs_dim, n_actions
        self.actor = _mlp(obs_dim, hidden_sizes, n_actions, out_gain=0.01)   # near-uniform initial policy
        self.critic = _mlp(obs_dim, hidden_sizes, 1, out_gain=1.0)

    def _check(self, x: torch.Tensor) -> None:
        if x.shape[-1] != self.obs_dim:
            raise ValueError(f"expected {self.obs_dim} input features, got {x.shape[-1]}")

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        self._check(x)
        return self.actor(x)

    def value(self, x: torch.Tensor) -> torch.Tensor:
        self._check(x)
        return self.critic(x).squeeze(-1)


def masked_log_probs(logits: torch.Tensor, masks: torch.Tensor | None) -> torch.Tensor:
    """log pi(a|s); invalid actions get -inf (probability exactly 0). ``masks=None`` leaves all actions in."""
    if masks is not None:
        logits = logits.masked_fill(~masks, float("-inf"))
    return torch.log_softmax(logits, dim=-1)


def entropy_of(log_probs: torch.Tensor) -> torch.Tensor:
    """Entropy per row; terms with probability 0 contribute 0 (no 0 * -inf)."""
    probs = log_probs.exp()
    return -(probs * torch.where(probs > 0, log_probs, torch.zeros_like(log_probs))).sum(dim=-1)


def compute_gae(rewards: np.ndarray, values: np.ndarray, next_values: np.ndarray, episode_ends: np.ndarray,
                gamma: float, gae_lambda: float) -> tuple[np.ndarray, np.ndarray]:
    """Generalised advantage estimation over one rollout.

    next_values[t] is the bootstrap value of the state after step t: V(s_{t+1}); 0 after a terminal step; the
    value of the final observation after a bootstrapped truncation. episode_ends[t] stops the recursion.
    Returns (advantages, returns = advantages + values).
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float64)
    gae = 0.0
    for t in range(T - 1, -1, -1):
        delta = rewards[t] + gamma * next_values[t] - values[t]
        gae = delta + gamma * gae_lambda * (0.0 if episode_ends[t] else gae)
        advantages[t] = gae
    return advantages, advantages + values


class PPOAgent:
    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes: list[int], learning_rate: float, gamma: float,
                 gae_lambda: float, clip_epsilon: float, entropy_coefficient: float, value_coefficient: float,
                 max_grad_norm: float, rollout_steps: int, update_epochs: int, minibatch_size: int,
                 use_action_mask: bool = True, normalize_advantages: bool = True,
                 truncation_is_terminal: bool = False, seed: int = 42, device: str = "cpu") -> None:
        self.obs_dim, self.n_actions, self.hidden_sizes = obs_dim, n_actions, list(hidden_sizes)
        self.learning_rate, self.gamma, self.gae_lambda = learning_rate, gamma, gae_lambda
        self.clip_epsilon, self.entropy_coefficient = clip_epsilon, entropy_coefficient
        self.value_coefficient, self.max_grad_norm = value_coefficient, max_grad_norm
        self.rollout_steps, self.update_epochs, self.minibatch_size = rollout_steps, update_epochs, minibatch_size
        self.use_action_mask, self.normalize_advantages = use_action_mask, normalize_advantages
        self.truncation_is_terminal = truncation_is_terminal
        self.seed, self.device = seed, resolve_device(device)

        torch.manual_seed(derive_seed(seed, "ppo", "init"))
        self.model = ActorCritic(obs_dim, n_actions, hidden_sizes).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, eps=1e-5)
        self.sample_gen = torch.Generator().manual_seed(derive_seed(seed, "ppo", "sampling"))
        self.shuffle_rng = make_rng(derive_seed(seed, "ppo", "minibatch"))
        self.env_steps = self.updates = self.gradient_steps = self.episodes_trained = 0

    @classmethod
    def from_config(cls, cfg: PPOConfig, obs_dim: int = 30, n_actions: int = 18, **overrides) -> "PPOAgent":
        params = dict(hidden_sizes=cfg.hidden_sizes, learning_rate=cfg.learning_rate, gamma=cfg.gamma,
                      gae_lambda=cfg.gae_lambda, clip_epsilon=cfg.clip_epsilon,
                      entropy_coefficient=cfg.entropy_coefficient, value_coefficient=cfg.value_coefficient,
                      max_grad_norm=cfg.max_grad_norm, rollout_steps=cfg.rollout_steps,
                      update_epochs=cfg.update_epochs, minibatch_size=cfg.minibatch_size,
                      use_action_mask=cfg.use_action_mask, normalize_advantages=cfg.normalize_advantages,
                      truncation_is_terminal=cfg.truncation_is_terminal, seed=cfg.seed, device=cfg.device)
        params.update(overrides)
        return cls(obs_dim, n_actions, **params)

    # ------------------------------------------------------------ acting
    def _obs(self, obs) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.shape != (self.obs_dim,):
            raise ValueError(f"observation must have shape ({self.obs_dim},), got {obs.shape}")
        return obs

    def _mask(self, mask) -> np.ndarray | None:
        if mask is None or not self.use_action_mask:
            return None
        mask = np.asarray(mask, dtype=bool)
        if not mask.any():
            raise ValueError("action mask has no valid action")
        return mask

    def distribution(self, obs, mask=None) -> np.ndarray:
        """Action probabilities for one observation (zeros for masked-out actions)."""
        m = self._mask(mask)
        with torch.no_grad():
            x = torch.as_tensor(self._obs(obs), device=self.device).unsqueeze(0)
            lp = masked_log_probs(self.model.logits(x), None if m is None else torch.as_tensor(m).unsqueeze(0))
        return lp.exp().squeeze(0).cpu().numpy().astype(np.float64)

    def value(self, obs) -> float:
        with torch.no_grad():
            return float(self.model.value(torch.as_tensor(self._obs(obs), device=self.device).unsqueeze(0)).item())

    def act(self, obs, mask=None) -> tuple[int, float, float, float]:
        """Sample an action. Returns (action, log-probability, value estimate, policy entropy)."""
        m = self._mask(mask)
        with torch.no_grad():
            x = torch.as_tensor(self._obs(obs), device=self.device).unsqueeze(0)
            lp = masked_log_probs(self.model.logits(x), None if m is None else torch.as_tensor(m).unsqueeze(0))
            action = int(torch.multinomial(lp.exp().squeeze(0).cpu(), 1, generator=self.sample_gen).item())
            return action, float(lp[0, action].item()), float(self.model.value(x).item()), float(entropy_of(lp).item())

    def predict(self, obs, mask=None) -> int:
        """Deterministic action: the most probable valid action; exact ties -> lowest index."""
        return int(np.argmax(self.distribution(obs, mask)))

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor, masks: torch.Tensor | None
                         ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lp = masked_log_probs(self.model.logits(obs), masks if self.use_action_mask else None)
        return lp.gather(1, actions.unsqueeze(1)).squeeze(1), entropy_of(lp), self.model.value(obs)

    # ----------------------------------------------------------- learning
    def update(self, obs: np.ndarray, actions: np.ndarray, old_log_probs: np.ndarray, advantages: np.ndarray,
               returns: np.ndarray, values: np.ndarray, masks: np.ndarray) -> dict:
        """PPO update on one rollout. Returns averaged diagnostics."""
        t = lambda x, dtype: torch.as_tensor(x, device=self.device, dtype=dtype)  # noqa: E731
        obs_t, act_t = t(obs, torch.float32), t(actions, torch.int64)
        old_lp, adv_t, ret_t = t(old_log_probs, torch.float32), t(advantages, torch.float32), t(returns, torch.float32)
        mask_t = t(masks, torch.bool)
        n = len(obs)
        stats = {k: [] for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction")}
        for _ in range(self.update_epochs):
            order = self.shuffle_rng.permutation(n)
            for start in range(0, n, self.minibatch_size):
                idx = torch.as_tensor(order[start:start + self.minibatch_size])
                new_lp, entropy, value = self.evaluate_actions(obs_t[idx], act_t[idx], mask_t[idx])
                adv = adv_t[idx]
                if self.normalize_advantages and len(idx) > 1:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                log_ratio = new_lp - old_lp[idx]
                ratio = log_ratio.exp()
                policy_loss = -torch.min(ratio * adv,
                                         torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv).mean()
                value_loss = 0.5 * (ret_t[idx] - value).pow(2).mean()
                loss = policy_loss + self.value_coefficient * value_loss - self.entropy_coefficient * entropy.mean()
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.gradient_steps += 1
                with torch.no_grad():
                    stats["policy_loss"].append(policy_loss.item())
                    stats["value_loss"].append(value_loss.item())
                    stats["entropy"].append(entropy.mean().item())
                    stats["approx_kl"].append(((ratio - 1) - log_ratio).mean().item())
                    stats["clip_fraction"].append(((ratio - 1).abs() > self.clip_epsilon).float().mean().item())
        self.updates += 1
        out = {k: float(np.mean(v)) for k, v in stats.items()}
        if not all(np.isfinite(v) for v in out.values()):
            raise FloatingPointError(f"non-finite PPO statistics at update {self.updates}: {out}")
        var = float(np.var(returns))
        out["explained_variance"] = float(1 - np.var(returns - values) / var) if var > 0 else float("nan")
        out["mean_return"] = float(np.mean(returns))
        out["mean_value"] = float(np.mean(values))
        return out

    def learn(self, env, total_timesteps: int, env_seed_fn=None, on_episode_end=None, on_update=None) -> None:
        """Train for ``total_timesteps // rollout_steps`` whole rollouts on any Gymnasium env with Discrete actions."""
        T = self.rollout_steps
        obs_buf = np.zeros((T, self.obs_dim), dtype=np.float32)
        mask_buf = np.ones((T, self.n_actions), dtype=bool)
        act_buf, lp_buf = np.zeros(T, dtype=np.int64), np.zeros(T)
        rew_buf, val_buf, next_val = np.zeros(T), np.zeros(T), np.zeros(T)
        end_buf = np.zeros(T, dtype=bool)

        def reset():
            seed = env_seed_fn(self.episodes_trained) if env_seed_fn else None
            o, info = env.reset(seed=seed)
            return self._obs(o), self._env_mask(env, info), seed, {"steps": 0, "reward": 0.0, "invalid": 0,
                                                                  "entropy": 0.0, "value": 0.0, "temp": 0.0}

        obs, mask, ep_seed, ep = reset()
        for _ in range(total_timesteps // T):
            for t in range(T):
                action, lp, value, ent = self.act(obs, mask)
                next_obs, reward, terminated, truncated, info = env.step(action)
                next_obs = self._obs(next_obs)
                obs_buf[t], act_buf[t], lp_buf[t], rew_buf[t], val_buf[t] = obs, action, lp, reward, value
                if mask is not None:
                    mask_buf[t] = mask
                done = bool(terminated or truncated)
                end_buf[t] = done
                self.env_steps += 1
                ep["steps"] += 1
                ep["reward"] += float(reward)
                ep["invalid"] += not info.get("action_valid", True)
                ep["entropy"] += ent
                ep["value"] += value
                ep["temp"] += info.get("avg_temperature", 0.0)
                if done:
                    bootstrap = truncated and not terminated and not self.truncation_is_terminal
                    next_val[t] = self.value(next_obs) if bootstrap else 0.0
                    self.episodes_trained += 1
                    if on_episode_end:
                        on_episode_end(env, info, ep, ep_seed)
                    obs, mask, ep_seed, ep = reset()
                else:
                    obs, mask = next_obs, self._env_mask(env, None)
            # Bootstrap values: V of the next stored state, except at episode ends (set above) and at the
            # last step of a rollout that stops mid-episode (V of the current observation).
            for t in range(T):
                if not end_buf[t]:
                    next_val[t] = val_buf[t + 1] if t < T - 1 else self.value(obs)
            adv, ret = compute_gae(rew_buf, val_buf, next_val, end_buf, self.gamma, self.gae_lambda)
            stats = self.update(obs_buf, act_buf, lp_buf, adv, ret, val_buf.copy(), mask_buf)
            if on_update:
                on_update(stats)

    @staticmethod
    def _env_mask(env, info: dict | None) -> np.ndarray | None:
        if info is not None and "action_mask" in info:
            return np.asarray(info["action_mask"], dtype=bool)
        masks = getattr(env, "action_masks", None)
        return None if masks is None else np.asarray(masks(), dtype=bool)

    # -------------------------------------------------------- persistence
    def hyperparameters(self) -> dict:
        return {"hidden_sizes": self.hidden_sizes, "learning_rate": self.learning_rate, "gamma": self.gamma,
                "gae_lambda": self.gae_lambda, "clip_epsilon": self.clip_epsilon,
                "entropy_coefficient": self.entropy_coefficient, "value_coefficient": self.value_coefficient,
                "max_grad_norm": self.max_grad_norm, "rollout_steps": self.rollout_steps,
                "update_epochs": self.update_epochs, "minibatch_size": self.minibatch_size,
                "use_action_mask": self.use_action_mask, "normalize_advantages": self.normalize_advantages,
                "truncation_is_terminal": self.truncation_is_terminal, "seed": self.seed}

    def save(self, directory: Path, metadata: dict | None = None) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        counters = {"env_steps": self.env_steps, "updates": self.updates, "gradient_steps": self.gradient_steps,
                    "episodes_trained": self.episodes_trained}
        torch.save({"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(), "counters": counters,
                    "obs_dim": self.obs_dim, "n_actions": self.n_actions,
                    "hyperparameters": self.hyperparameters()}, directory / CHECKPOINT)
        payload = {"algorithm": "ppo", "obs_dim": self.obs_dim, "n_actions": self.n_actions,
                   "architecture": {"actor": repr(self.model.actor), "critic": repr(self.model.critic)},
                   "device": str(self.device), **counters, "hyperparameters": self.hyperparameters(),
                   "checkpoint": CHECKPOINT, **(metadata or {})}
        (directory / META_FILE).write_text(json.dumps(payload, indent=1), encoding="utf-8")
        return directory / META_FILE

    @classmethod
    def load(cls, directory: Path, device: str = "cpu") -> "PPOAgent":
        directory = Path(directory)
        ckpt = torch.load(directory / CHECKPOINT, map_location="cpu", weights_only=True)
        agent = cls(ckpt["obs_dim"], ckpt["n_actions"], device=device, **ckpt["hyperparameters"])
        agent.model.load_state_dict(ckpt["model"])
        agent.optimizer.load_state_dict(ckpt["optimizer"])
        for k, v in ckpt["counters"].items():
            setattr(agent, k, v)
        agent.metadata = json.loads((directory / META_FILE).read_text(encoding="utf-8"))
        return agent


class PPOController(Controller):
    """PPO policy for the shared runner. Deterministic (most probable valid action) by default; with
    ``deterministic=False`` it samples from the policy using a generator reseeded per episode.
    Optionally records observations, masks and actions for analysis."""

    name = "ppo"

    def __init__(self, agent: PPOAgent, config: Config | None = None, deterministic: bool = True,
                 record: bool = False) -> None:
        super().__init__(config)
        self.agent, self.deterministic, self.record = agent, deterministic, record
        self.gen = torch.Generator()
        self.action_counts = np.zeros(agent.n_actions, dtype=np.int64)
        self.observations: list[np.ndarray] = []
        self.masks: list[np.ndarray] = []
        self.actions: list[int] = []

    def reset(self, seed: int | None = None) -> None:
        self.gen.manual_seed(derive_seed(self.agent.seed if seed is None else seed, "ppo", "evaluation"))

    def act(self, observation, action_mask=None) -> int:
        probs = self.agent.distribution(observation, action_mask)
        if self.deterministic:
            action = int(np.argmax(probs))
        else:
            action = int(torch.multinomial(torch.as_tensor(probs), 1, generator=self.gen).item())
        self.action_counts[action] += 1
        if self.record:
            self.observations.append(np.asarray(observation, dtype=np.float32))
            self.masks.append(np.asarray(action_mask, dtype=bool))
            self.actions.append(action)
        return action


# --------------------------------------------------------------------------- #
# Data-centre training
# --------------------------------------------------------------------------- #
EPISODE_FIELDS = ("episode", "env_seed", "start_hour", "total_reward", "energy_kwh", "it_energy_kwh",
                  "cooling_energy_kwh", "carbon_kg", "cost_inr", "sla_failures", "completed_jobs",
                  "critical_rack_minutes", "warning_or_worse_rack_minutes", "peak_temperature_c", "invalid_actions",
                  "invalid_action_rate", "mean_entropy", "mean_value", "env_steps", "updates")
UPDATE_FIELDS = ("update", "env_steps", "gradient_steps", "policy_loss", "value_loss", "entropy", "approx_kl",
                 "clip_fraction", "explained_variance", "mean_return", "mean_value")


def train_data_centre(agent: PPOAgent, config: Config, total_timesteps: int, episode_minutes: int,
                      randomize_start_hour: bool, log_every: int = 10) -> tuple[list[dict], list[dict]]:
    """Returns (per-episode rows, per-update rows). Episode e uses env seed derive_seed(seed, 'ppo', 'episode', e)."""
    env = DataCenterEnv(representation="features", episode_minutes=episode_minutes,
                        randomize_start_hour=randomize_start_hour, config=config)
    assert env.observation_space.shape == (agent.obs_dim,) and env.action_space.n == agent.n_actions
    episodes, updates, start = [], [], time.perf_counter()

    def on_episode_end(env_, info, ep, seed):
        metrics = episode_metrics(env_, info, ep["temp"], seed)
        row = {"episode": len(episodes), "env_seed": seed, "start_hour": env_.dc.start_hour, **metrics,
               "invalid_actions": ep["invalid"], "invalid_action_rate": ep["invalid"] / ep["steps"],
               "mean_entropy": ep["entropy"] / ep["steps"], "mean_value": ep["value"] / ep["steps"],
               "env_steps": agent.env_steps, "updates": agent.updates}
        episodes.append({k: row[k] for k in EPISODE_FIELDS})

    def on_update(stats):
        updates.append({"update": agent.updates, "env_steps": agent.env_steps, "gradient_steps": agent.gradient_steps,
                        **{k: stats[k] for k in UPDATE_FIELDS if k in stats}})
        if log_every and (agent.updates % log_every == 0):
            recent = episodes[-10:]
            logger.info("%s update %d  steps %d  reward(last %d ep) %.1f  entropy %.3f  kl %.4f  clip %.2f  "
                        "v_loss %.2f  EV %.2f  %.0f s", "mask" if agent.use_action_mask else "no-mask", agent.updates,
                        agent.env_steps, len(recent),
                        np.mean([r["total_reward"] for r in recent]) if recent else float("nan"),
                        stats["entropy"], stats["approx_kl"], stats["clip_fraction"], stats["value_loss"],
                        stats["explained_variance"], time.perf_counter() - start)

    agent.learn(env, total_timesteps, env_seed_fn=lambda e: derive_seed(agent.seed, "ppo", "episode", e),
                on_episode_end=on_episode_end, on_update=on_update)
    return episodes, updates


def write_csv(rows: list[dict], fields: tuple[str, ...], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    cfg = get_config()
    p = cfg.rl.ppo
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-mask", action="store_true", help="masking ablation (use_action_mask = False)")
    parser.add_argument("--timesteps", type=int, default=p.total_timesteps)
    parser.add_argument("--minutes", type=int, default=p.episode_minutes)
    parser.add_argument("--seed", type=int, default=p.seed)
    parser.add_argument("--tag", type=str, default=None, help="output name under models/ and experiments/")
    args = parser.parse_args()
    tag = args.tag or ("ppo_nomask" if args.no_mask else "ppo")
    out = cfg.paths.models / tag
    ep_csv, up_csv = cfg.paths.experiments / f"{tag}_training.csv", cfg.paths.experiments / f"{tag}_updates.csv"
    design = args.timesteps == p.total_timesteps and args.minutes == p.episode_minutes
    if not design:
        logger.warning("%d timesteps x %d-min episodes differs from the design protocol (%d, %d)",
                       args.timesteps, args.minutes, p.total_timesteps, p.episode_minutes)

    set_global_seed(args.seed)
    configure_torch(p.torch_threads)
    agent = PPOAgent.from_config(p, seed=args.seed, use_action_mask=not args.no_mask)
    started = time.perf_counter()
    episodes, updates = train_data_centre(agent, cfg, args.timesteps, args.minutes, p.randomize_start_hour)
    seconds = time.perf_counter() - started
    write_csv(episodes, EPISODE_FIELDS, ep_csv)
    write_csv(updates, UPDATE_FIELDS, up_csv)
    agent.save(out, metadata={"training": {
        "requested_timesteps": args.timesteps, "total_env_steps": agent.env_steps, "rollouts": agent.updates,
        "gradient_steps": agent.gradient_steps, "episodes_completed": len(episodes),
        "episode_minutes": args.minutes, "design_protocol": design, "seconds": seconds,
        "device": str(agent.device), "torch_threads": p.torch_threads, "torch_version": torch.__version__,
        "randomize_start_hour": p.randomize_start_hour,
        "env_seed_strategy": "episode e: derive_seed(seed, 'ppo', 'episode', e)",
        "rng_streams": "init / sampling / minibatch: derive_seed(seed, 'ppo', <name>)",
        "episode_csv": str(ep_csv), "update_csv": str(up_csv)}})
    print(f"PPO [{tag}]: {agent.env_steps} steps ({agent.updates} rollouts x {agent.rollout_steps}), "
          f"{agent.gradient_steps} gradient steps, {len(episodes)} complete episodes, {seconds:.0f} s on {agent.device}")
    logger.info("wrote %s, %s, %s", out, ep_csv, up_csv)


if __name__ == "__main__":
    main()
