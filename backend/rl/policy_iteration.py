"""Tabular Policy Iteration (Step 7).

The algorithm (Sutton & Barto §4.3) runs on any ``TabularMDP``: the toy MDPs in the tests, or the
empirical model of the 144-state aggregated data-centre MDP from ``backend.rl.tabular_model``.

    initialise   pi_0(s) = lowest-index admissible action
    repeat
      evaluate   V(s) <- sum_s' P(s'|s,pi(s)) [R(s,pi(s),s') + gamma V(s')]      (synchronous sweeps,
                 until max_s |V_new(s) - V_old(s)| < theta, or max_evaluation_sweeps; warm-started)
      improve    Q(s,a) = sum_s' P(s'|s,a) [R(s,a,s') + gamma V(s')]              for admissible a
                 pi'(s) = argmax_a Q(s,a); Q-values within tie_tolerance of the maximum are ties.
                 Ties keep the current action if it is among them (so the loop cannot cycle between
                 equally good policies), otherwise the lowest action index wins.
    until pi' == pi (stable) or max_iterations

Train on the data-centre model and save to ``models/policy_iteration/``:

    python -m backend.rl.policy_iteration
    python -m backend.rl.policy_iteration --samples 100 --seed 42
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from backend.config import Config, PolicyIterationConfig, get_config
from backend.rl.actions import N_TABULAR_ACTIONS, CoolingCommand, decode_tabular_action
from backend.rl.baselines import Controller
from backend.rl.state_encoder import LEVEL_NAMES, TabularStateEncoder
from backend.rl.tabular_model import (
    EmpiricalModel,
    TabularMDP,
    build_empirical_model,
    greedy_actions,
    load_model,
    model_hash,
    q_values,
    save_model,
    tied_best,
)
from backend.utils.logging_utils import get_logger
from backend.utils.seeding import set_global_seed

logger = get_logger("policy_iteration")

POLICY_FILE = "policy.json"
MODEL_FILE = "model.npz"


@dataclass
class IterationRecord:
    iteration: int                 # outer policy-iteration number, from 1
    evaluation_sweeps: int
    evaluation_converged: bool
    final_value_delta: float       # max_s |V_new - V_old| at the last sweep
    sweep_deltas: list[float]      # max value change of every evaluation sweep
    policy_changes: int            # states whose action changed in the improvement step
    policy_stable: bool


@dataclass
class TrainingResult:
    converged: bool
    iterations: int
    history: list[IterationRecord] = field(default_factory=list)
    seconds: float = 0.0


class PolicyIteration:
    def __init__(self, mdp: TabularMDP, gamma: float, theta: float, max_iterations: int,
                 max_evaluation_sweeps: int, tie_tolerance: float = 1e-9) -> None:
        if not 0 < gamma < 1:
            raise ValueError("gamma must be in (0, 1)")
        self.mdp = mdp
        self.gamma = gamma
        self.theta = theta
        self.max_iterations = max_iterations
        self.max_evaluation_sweeps = max_evaluation_sweeps
        self.tie_tolerance = tie_tolerance
        self.policy = self.initialize_policy()
        self.V = np.zeros(mdp.n_states)
        self.result: TrainingResult | None = None

    @classmethod
    def from_config(cls, mdp: TabularMDP, cfg: PolicyIterationConfig) -> "PolicyIteration":
        return cls(mdp, cfg.gamma, cfg.theta, cfg.max_iterations, cfg.max_evaluation_sweeps, cfg.tie_tolerance)

    # ---------------------------------------------------------------- algorithm
    def initialize_policy(self) -> np.ndarray:
        """Deterministic: the lowest-index admissible action in every state."""
        return np.argmax(self.mdp.admissible, axis=1).astype(np.int64)

    def policy_evaluation(self, policy: np.ndarray, V: np.ndarray | None = None
                          ) -> tuple[np.ndarray, list[float], bool]:
        """Iterative evaluation of a deterministic policy. Returns (V, per-sweep max deltas, converged)."""
        states = np.arange(self.mdp.n_states)
        P_pi = self.mdp.P[states, policy]                            # (S, S)
        R_pi = np.einsum("st,st->s", P_pi, self.mdp.R[states, policy])  # expected one-step reward
        V = np.zeros(self.mdp.n_states) if V is None else V.astype(np.float64, copy=True)
        deltas: list[float] = []
        for _ in range(self.max_evaluation_sweeps):
            V_new = R_pi + self.gamma * P_pi @ V
            delta = float(np.max(np.abs(V_new - V)))
            V = V_new
            deltas.append(delta)
            if delta < self.theta:
                return V, deltas, True
        return V, deltas, False

    def q_values(self, V: np.ndarray | None = None) -> np.ndarray:
        """Q(s,a) for every pair; -inf for inadmissible actions."""
        return q_values(self.mdp, self.V if V is None else V, self.gamma)

    def policy_improvement(self, V: np.ndarray, policy: np.ndarray) -> tuple[np.ndarray, int]:
        """Greedy policy over admissible actions. Returns (new policy, number of states changed)."""
        Q = self.q_values(V)
        ties = tied_best(Q, self.tie_tolerance)
        keep = ties[np.arange(len(policy)), policy]
        new_policy = np.where(keep, policy, greedy_actions(Q, self.tie_tolerance))
        return new_policy, int(np.sum(new_policy != policy))

    def train(self) -> TrainingResult:
        start = time.perf_counter()
        policy, V = self.initialize_policy(), np.zeros(self.mdp.n_states)
        history: list[IterationRecord] = []
        converged = False
        for iteration in range(1, self.max_iterations + 1):
            V, deltas, eval_converged = self.policy_evaluation(policy, V)
            if not eval_converged:
                logger.warning("iteration %d: evaluation did not reach theta in %d sweeps (delta %.2e)",
                               iteration, len(deltas), deltas[-1])
            new_policy, changes = self.policy_improvement(V, policy)
            stable = changes == 0
            history.append(IterationRecord(iteration, len(deltas), eval_converged, deltas[-1], deltas,
                                           changes, stable))
            logger.info("iteration %d: %d sweeps, final delta %.2e, %d actions changed",
                        iteration, len(deltas), deltas[-1], changes)
            policy = new_policy
            if stable:
                converged = True
                break
        self.policy, self.V = policy, V
        if not np.isfinite(self.V).all():
            raise FloatingPointError("value function is not finite")
        self.result = TrainingResult(converged, len(history), history, time.perf_counter() - start)
        return self.result

    # ------------------------------------------------------------------ queries
    def get_policy(self) -> np.ndarray:
        return self.policy.copy()

    def get_value_function(self) -> np.ndarray:
        return self.V.copy()

    def predict(self, state: int) -> int:
        return int(self.policy[int(state)])


# --------------------------------------------------------------------------- #
# Controller + persistence for the data-centre policy
# --------------------------------------------------------------------------- #
class PolicyIterationController(Controller):
    """Runs a learned tabular policy in the tabular environment; records visited states and actions."""

    name = "policy_iteration"

    def __init__(self, policy: np.ndarray, config: Config | None = None) -> None:
        super().__init__(config)
        self.policy = np.asarray(policy, dtype=np.int64)
        self.state_visits = np.zeros(len(self.policy), dtype=np.int64)
        self.action_counts = np.zeros(N_TABULAR_ACTIONS, dtype=np.int64)

    def act(self, observation, action_mask: np.ndarray | None = None) -> int:
        state = int(observation)
        action = int(self.policy[state])
        self.state_visits[state] += 1
        self.action_counts[action] += 1
        return action


def describe_action(action: int) -> str:
    placement, cooling = decode_tabular_action(action)
    return f"{placement.name.lower()} + {CoolingCommand(cooling).name.lower()} cooling"


def space_definitions(config: Config) -> dict:
    encoder = TabularStateEncoder(config.discretisation)
    return {
        "state_space": {
            "n_states": encoder.n_states,
            "factors": list(encoder.FACTORS),
            "levels": {f: list(LEVEL_NAMES[f]) for f in encoder.FACTORS},
            "edges": {f: encoder.edges[f].tolist() for f in encoder.FACTORS},
            "encoding": "np.ravel_multi_index(levels, shape), C order (backend/rl/state_encoder.py)",
        },
        "action_space": {
            "n_actions": N_TABULAR_ACTIONS,
            "encoding": "a = placement * 3 + cooling (backend/rl/actions.py)",
            "actions": [describe_action(a) for a in range(N_TABULAR_ACTIONS)],
        },
    }


def save(pi: PolicyIteration, model: EmpiricalModel, config: Config, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    save_model(model, directory / MODEL_FILE)
    result = pi.result
    payload = {
        "algorithm": "policy_iteration",
        "gamma": pi.gamma,
        "hyperparameters": config.rl.policy_iteration.model_dump(),
        "model": {**model.metadata, "file": MODEL_FILE, "model_hash": model_hash(model),
                  "modelled_states": np.flatnonzero(model.modelled).tolist()},
        **space_definitions(config),
        "policy": pi.policy.tolist(),
        "value_function": pi.V.tolist(),
        "admissible": pi.mdp.admissible.astype(int).tolist(),
        "training": {"converged": result.converged, "iterations": result.iterations,
                     "seconds": result.seconds, "history": [asdict(h) for h in result.history]}
                    if result else None,
    }
    path = directory / POLICY_FILE
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return path


@dataclass
class LoadedPolicy:
    policy: np.ndarray
    value_function: np.ndarray
    payload: dict
    model: EmpiricalModel

    def predict(self, state: int) -> int:
        return int(self.policy[int(state)])


def load(directory: Path) -> LoadedPolicy:
    payload = json.loads((directory / POLICY_FILE).read_text(encoding="utf-8"))
    model = load_model(directory / payload["model"]["file"], payload["model"])
    stored = payload["model"].get("model_hash")
    if stored is not None and stored != model_hash(model):
        raise ValueError(f"{directory / payload['model']['file']} does not match the model_hash in {POLICY_FILE}")
    return LoadedPolicy(np.asarray(payload["policy"], dtype=np.int64),
                        np.asarray(payload["value_function"], dtype=np.float64), payload, model)


def train_policy_iteration(config: Config | None = None, seed: int | None = None,
                           samples: int | None = None) -> tuple[PolicyIteration, EmpiricalModel]:
    config = config or get_config()
    model = build_empirical_model(config, seed=seed, samples_per_state_action=samples)
    pi = PolicyIteration.from_config(model.to_mdp(), config.rl.policy_iteration)
    pi.train()
    return pi, model


def main() -> None:
    cfg = get_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=cfg.project.seed)
    parser.add_argument("--samples", type=int, default=cfg.rl.policy_iteration.model_samples_per_state_action)
    parser.add_argument("--out", type=Path, default=cfg.paths.models / "policy_iteration")
    args = parser.parse_args()

    set_global_seed(args.seed)
    started = time.perf_counter()
    pi, model = train_policy_iteration(cfg, args.seed, args.samples)
    path = save(pi, model, cfg, args.out)
    r = pi.result
    print(f"Policy Iteration {'converged' if r.converged else 'did NOT converge'} in {r.iterations} iterations "
          f"({sum(h.evaluation_sweeps for h in r.history)} evaluation sweeps); "
          f"{int(model.modelled.sum())} modelled states; total {time.perf_counter() - started:.0f} s")
    logger.info("wrote %s", path)


if __name__ == "__main__":
    main()
