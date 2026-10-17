"""Tabular Q-Learning (Step 9).

Model-free: the agent learns only from ``env.reset()`` / ``env.step()`` experience in the tabular environment
(144 states x 9 macro-actions). It never reads the Step 7/8 empirical model, and Q starts at 0 everywhere.

Per step, with s the abstract state and M(s) the environment's *actual* action mask (``env.action_masks()``,
computed from the concrete simulator state, so it can differ between visits to the same abstract state):

    a  ~ epsilon-greedy over M(s): uniform over valid actions with probability epsilon, else greedy
    s', r = env.step(a)
    target = r                                          if the transition is terminal
           = r + gamma * max_{a' in M(s')} Q(s', a')    otherwise
    Q(s,a) <- Q(s,a) + alpha * (target - Q(s,a))

The data-centre environment never terminates. The end of an episode is a time-limit *truncation*, so its last
transition is still bootstrapped.

Greedy choice: the highest Q among valid actions; values within ``tie_tolerance`` of the maximum are ties, and
the lowest index wins (as in PI/VI). Q-values of invalid actions are stored but never selected, and never used
in a target.

    python -m backend.rl.q_learning                     # full protocol from config.yaml -> models/q_learning/
    python -m backend.rl.q_learning --episodes 20 --minutes 240 --out models/q_learning_dev
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.config import Config, QLearningConfig, get_config
from backend.evaluation.runner import episode_metrics
from backend.rl.actions import N_TABULAR_ACTIONS, CoolingCommand, TabularPlacement, encode_tabular_action
from backend.rl.baselines import Controller
from backend.rl.environment import DataCenterEnv
from backend.rl.policy_iteration import space_definitions
from backend.utils.logging_utils import get_logger
from backend.utils.seeding import derive_seed, make_rng, set_global_seed

logger = get_logger("q_learning")

Q_FILE = "q_table.npz"
META_FILE = "agent.json"
# Delay + maintain cooling is valid in every concrete state; used for states never visited in training.
FALLBACK_ACTION = encode_tabular_action(TabularPlacement.DELAY, CoolingCommand.MAINTAIN)


@dataclass
class StepResult:
    action: int
    explored: bool
    reward: float
    q_change: float        # signed change applied to Q(s, a)
    info: dict


class QLearning:
    def __init__(self, n_states: int, n_actions: int, learning_rate: float, gamma: float, epsilon_start: float,
                 epsilon_min: float, epsilon_decay: float, tie_tolerance: float = 1e-4, seed: int = 42) -> None:
        if not 0 < learning_rate <= 1 or not 0 <= gamma < 1:
            raise ValueError("learning_rate must be in (0, 1] and gamma in [0, 1)")
        self.n_states, self.n_actions = n_states, n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_start, self.epsilon_min, self.epsilon_decay = epsilon_start, epsilon_min, epsilon_decay
        self.tie_tolerance = tie_tolerance
        self.seed = seed
        self.initialize()

    @classmethod
    def from_config(cls, cfg: QLearningConfig, n_states: int = 144, n_actions: int = N_TABULAR_ACTIONS,
                    seed: int | None = None) -> "QLearning":
        return cls(n_states, n_actions, cfg.learning_rate, cfg.gamma, cfg.epsilon_start, cfg.epsilon_min,
                   cfg.epsilon_decay, cfg.tie_tolerance, cfg.seed if seed is None else seed)

    def initialize(self) -> None:
        """Q = 0 everywhere, empty counters, exploration RNG derived from the training seed."""
        self.q_table = np.zeros((self.n_states, self.n_actions))
        self.visit_counts = np.zeros((self.n_states, self.n_actions), dtype=np.int64)   # times (s, a) was taken
        self.valid_counts = np.zeros((self.n_states, self.n_actions), dtype=np.int64)   # times a was valid in s
        self.state_counts = np.zeros(self.n_states, dtype=np.int64)                     # decision steps in s
        self.rng = make_rng(derive_seed(self.seed, "q-learning", "exploration"))
        self.episodes_trained = 0

    # ------------------------------------------------------------ exploration
    def epsilon_at(self, episode: int) -> float:
        """Per-episode schedule: max(epsilon_min, epsilon_start * epsilon_decay ** episode)."""
        return max(self.epsilon_min, self.epsilon_start * self.epsilon_decay ** episode)

    def _valid(self, mask: np.ndarray | None) -> np.ndarray:
        if mask is None:
            return np.ones(self.n_actions, dtype=bool)
        mask = np.asarray(mask, dtype=bool)
        if not mask.any():
            raise ValueError("action mask has no valid action")
        return mask

    def greedy_action(self, state: int, mask: np.ndarray | None = None) -> int:
        valid = self._valid(mask)
        q = np.where(valid, self.q_table[int(state)], -np.inf)
        return int(np.argmax(q >= q.max() - self.tie_tolerance))

    def select_action(self, state: int, mask: np.ndarray | None, epsilon: float) -> tuple[int, bool]:
        """Epsilon-greedy over the valid actions. Returns (action, explored)."""
        valid = self._valid(mask)
        if epsilon > 0 and self.rng.random() < epsilon:
            return int(self.rng.choice(np.flatnonzero(valid))), True
        return self.greedy_action(state, valid), False

    # ----------------------------------------------------------------- update
    def masked_max(self, state: int, mask: np.ndarray | None) -> float:
        """max over valid actions of Q(state, a); invalid actions cannot influence a target."""
        return float(np.max(self.q_table[int(state)][self._valid(mask)]))

    def update(self, state: int, action: int, reward: float, next_state: int, next_mask: np.ndarray | None,
               terminal: bool) -> float:
        target = reward if terminal else reward + self.gamma * self.masked_max(next_state, next_mask)
        change = self.learning_rate * (target - self.q_table[state, action])
        self.q_table[state, action] += change
        return float(change)

    # --------------------------------------------------------------- episodes
    def learn_episode(self, env, epsilon: float, env_seed: int | None = None, options: dict | None = None) -> dict:
        """One episode of learning in any Gymnasium env with Discrete states (masks via ``action_masks()``)."""
        state, info = env.reset(seed=env_seed, options=options)
        mask = self._mask_of(env, info)
        q_before = self.q_table.copy()
        steps = explored = invalid = 0
        total_reward, abs_changes = 0.0, []
        temperature_sum = 0.0
        done = False
        while not done:
            state = int(state)
            self.state_counts[state] += 1
            if mask is not None:
                self.valid_counts[state] += mask
            action, was_explore = self.select_action(state, mask, epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)
            next_mask = None if terminated else self._mask_of(env, None)
            change = self.update(state, action, float(reward), int(next_state), next_mask, bool(terminated))
            self.visit_counts[state, action] += 1
            steps += 1
            explored += was_explore
            invalid += not info.get("action_valid", True)
            total_reward += float(reward)
            abs_changes.append(abs(change))
            temperature_sum += info.get("avg_temperature", 0.0)
            state, mask = next_state, next_mask
            done = bool(terminated or truncated)
        self.episodes_trained += 1
        changed = self.q_table != q_before
        return {"steps": steps, "epsilon": epsilon, "explore_steps": explored, "exploit_steps": steps - explored,
                "invalid_actions": invalid, "episode_reward": total_reward,
                "max_abs_q_update": float(max(abs_changes)), "mean_abs_q_update": float(np.mean(abs_changes)),
                "q_entries_changed": int(changed.sum()), "states_visited_so_far": int((self.state_counts > 0).sum()),
                "_last_info": info, "_temperature_sum": temperature_sum}

    @staticmethod
    def _mask_of(env, info: dict | None) -> np.ndarray | None:
        if info is not None and "action_mask" in info:
            return np.asarray(info["action_mask"], dtype=bool)
        masks = getattr(env, "action_masks", None) or getattr(getattr(env, "unwrapped", None), "action_masks", None)
        return None if masks is None else np.asarray(masks(), dtype=bool)

    # ---------------------------------------------------------------- queries
    def get_q_table(self) -> np.ndarray:
        return self.q_table.copy()

    def admissible(self) -> np.ndarray:
        """bool[s, a]: actions observed valid at least once in s during training (same rule as PI/VI's model).
        States never visited admit only the always-valid fallback action."""
        adm = self.valid_counts > 0
        adm[self.state_counts == 0] = False
        adm[self.state_counts == 0, FALLBACK_ACTION] = True
        return adm

    def get_policy(self) -> np.ndarray:
        """Policy table pi(s) = argmax over admissible a of Q(s, a); ties -> lowest index."""
        adm = self.admissible()
        return np.array([self.greedy_action(s, adm[s]) for s in range(self.n_states)], dtype=np.int64)

    def predict(self, state: int, mask: np.ndarray | None = None) -> int:
        """Greedy action (epsilon = 0). With the live mask: argmax over it; without: the policy table."""
        if mask is not None:
            return self.greedy_action(state, mask)
        return int(self.get_policy()[int(state)])

    # ------------------------------------------------------------- persistence
    def hyperparameters(self) -> dict:
        return {"learning_rate": self.learning_rate, "gamma": self.gamma, "epsilon_start": self.epsilon_start,
                "epsilon_min": self.epsilon_min, "epsilon_decay": self.epsilon_decay,
                "tie_tolerance": self.tie_tolerance, "seed": self.seed}

    def save(self, directory: Path, config: Config | None = None, metadata: dict | None = None) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / Q_FILE, q_table=self.q_table, visit_counts=self.visit_counts,
                            valid_counts=self.valid_counts, state_counts=self.state_counts)
        payload = {
            "algorithm": "q_learning",
            **self.hyperparameters(),
            "episodes_trained": self.episodes_trained,
            "final_epsilon": self.epsilon_at(max(0, self.episodes_trained - 1)),
            "initial_q": 0.0,
            "q_table_file": Q_FILE,
            "q_table_shape": list(self.q_table.shape),
            "policy": self.get_policy().tolist(),
            "admissible": self.admissible().astype(int).tolist(),
            **(space_definitions(config) if config is not None else {}),
            **(metadata or {}),
        }
        (directory / META_FILE).write_text(json.dumps(payload, indent=1), encoding="utf-8")
        return directory / META_FILE

    @classmethod
    def load(cls, directory: Path) -> "QLearning":
        directory = Path(directory)
        payload = json.loads((directory / META_FILE).read_text(encoding="utf-8"))
        with np.load(directory / payload["q_table_file"]) as data:
            q_table = data["q_table"]
            agent = cls(q_table.shape[0], q_table.shape[1], payload["learning_rate"], payload["gamma"],
                        payload["epsilon_start"], payload["epsilon_min"], payload["epsilon_decay"],
                        payload["tie_tolerance"], payload["seed"])
            agent.q_table = q_table.copy()
            agent.visit_counts = data["visit_counts"].copy()
            agent.valid_counts = data["valid_counts"].copy()
            agent.state_counts = data["state_counts"].copy()
        agent.episodes_trained = payload["episodes_trained"]
        agent.metadata = payload
        return agent


class QLearningController(Controller):
    """Greedy (epsilon = 0) Q-Learning policy for the shared runner, using the live action mask."""

    name = "q_learning"

    def __init__(self, agent: QLearning, config: Config | None = None) -> None:
        super().__init__(config)
        self.agent = agent
        self.epsilon = 0.0
        self.state_visits = np.zeros(agent.n_states, dtype=np.int64)
        self.action_counts = np.zeros(agent.n_actions, dtype=np.int64)

    def act(self, observation, action_mask: np.ndarray | None = None) -> int:
        state = int(observation)
        action, explored = self.agent.select_action(state, action_mask, self.epsilon)
        assert not explored
        self.state_visits[state] += 1
        self.action_counts[action] += 1
        return action


# --------------------------------------------------------------------------- #
# Data-centre training
# --------------------------------------------------------------------------- #
CSV_FIELDS = ("episode", "env_seed", "start_hour", "epsilon", "total_reward", "energy_kwh", "it_energy_kwh",
              "cooling_energy_kwh", "carbon_kg", "cost_inr", "sla_failures", "completed_jobs", "delayed_jobs",
              "critical_rack_minutes", "warning_or_worse_rack_minutes", "peak_temperature_c", "invalid_actions",
              "explore_steps", "exploit_steps", "max_abs_q_update", "mean_abs_q_update", "q_entries_changed",
              "states_visited_so_far", "state_action_pairs_visited_so_far")


def train_data_centre(agent: QLearning, config: Config, episodes: int, episode_minutes: int,
                      randomize_start_hour: bool, log_every: int = 50) -> list[dict]:
    """Train on the tabular data-centre environment. Episode e uses env seed derive_seed(seed, 'q-learning', 'episode', e)."""
    env = DataCenterEnv(representation="tabular", episode_minutes=episode_minutes,
                        randomize_start_hour=randomize_start_hour, config=config)
    rows = []
    start = time.perf_counter()
    for e in range(episodes):
        env_seed = derive_seed(agent.seed, "q-learning", "episode", e)
        stats = agent.learn_episode(env, agent.epsilon_at(e), env_seed)
        metrics = episode_metrics(env, stats.pop("_last_info"), stats.pop("_temperature_sum"), env_seed)
        row = {"episode": e, "env_seed": env_seed, "start_hour": env.dc.start_hour, **stats,
               **{k: metrics[k] for k in CSV_FIELDS if k in metrics},
               "state_action_pairs_visited_so_far": int((agent.visit_counts > 0).sum())}
        rows.append({k: row[k] for k in CSV_FIELDS})
        if log_every and ((e + 1) % log_every == 0 or e == episodes - 1):
            recent = rows[-log_every:]
            logger.info("episode %d/%d  eps %.3f  reward %.1f (last %d mean %.1f)  SLA %d  states %d  %.0f s",
                        e + 1, episodes, row["epsilon"], row["total_reward"], len(recent),
                        np.mean([r["total_reward"] for r in recent]), row["sla_failures"],
                        row["states_visited_so_far"], time.perf_counter() - start)
    return rows


def write_training_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    cfg = get_config()
    q = cfg.rl.q_learning
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=q.episodes)
    parser.add_argument("--minutes", type=int, default=q.episode_minutes)
    parser.add_argument("--seed", type=int, default=q.seed)
    parser.add_argument("--out", type=Path, default=cfg.paths.models / "q_learning")
    parser.add_argument("--csv", type=Path, default=cfg.paths.experiments / "q_learning_training.csv")
    args = parser.parse_args()
    full = args.episodes == q.episodes and args.minutes == q.episode_minutes
    if not full:
        logger.warning("development run: %d x %d min differs from the configured protocol %d x %d min",
                       args.episodes, args.minutes, q.episodes, q.episode_minutes)

    set_global_seed(args.seed)
    agent = QLearning.from_config(q, seed=args.seed)
    started = time.perf_counter()
    rows = train_data_centre(agent, cfg, args.episodes, args.minutes, q.randomize_start_hour)
    seconds = time.perf_counter() - started
    write_training_csv(rows, args.csv)
    agent.save(args.out, cfg, metadata={
        "training": {"episodes": args.episodes, "episode_minutes": args.minutes, "total_steps": args.episodes * args.minutes,
                     "randomize_start_hour": q.randomize_start_hour, "full_protocol": full, "seconds": seconds,
                     "env_seed_strategy": "episode e: derive_seed(seed, 'q-learning', 'episode', e)",
                     "exploration_rng": "derive_seed(seed, 'q-learning', 'exploration')",
                     "training_csv": str(args.csv)}})
    print(f"Q-Learning: {args.episodes} episodes x {args.minutes} min in {seconds:.0f} s; final epsilon "
          f"{agent.epsilon_at(args.episodes - 1):.4f}; states visited {int((agent.state_counts > 0).sum())}/144; "
          f"state-action pairs {int((agent.visit_counts > 0).sum())}/{agent.q_table.size}")
    logger.info("wrote %s and %s", args.out, args.csv)


if __name__ == "__main__":
    main()
