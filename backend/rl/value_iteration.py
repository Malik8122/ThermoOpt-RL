"""Tabular Value Iteration (Step 8).

Solves the same ``TabularMDP`` as Policy Iteration. For the data centre, that is the Step 7 empirical model
loaded from ``models/policy_iteration/model.npz``, which is never rebuilt or resampled here.

    initialise   V_0(s) = 0 for every state
    repeat       V_{k+1}(s) = max_{a in A(s)} sum_s' P(s'|s,a) [R(s,a,s') + gamma V_k(s')]   (synchronous)
    until        max_s |V_{k+1}(s) - V_k(s)| < theta, or max_iterations
    extract      pi(s) = argmax_{a in A(s)} Q(s,a) with Q from the final V; Q-values within tie_tolerance of
                 the maximum are ties, and the lowest action index wins

A(s) is the model's admissible-action set, the same as for Policy Iteration (``EmpiricalModel.to_mdp``).
There is no current policy to keep during extraction, so ties go straight to the lowest index.

    python -m backend.rl.value_iteration      # -> models/value_iteration/policy.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from backend.config import PROJECT_ROOT, Config, ValueIterationConfig, get_config
from backend.rl.policy_iteration import POLICY_FILE, PolicyIterationController, space_definitions
from backend.rl.tabular_model import (
    EmpiricalModel,
    TabularMDP,
    greedy_actions,
    load_model,
    model_hash,
    q_values,
)
from backend.utils.logging_utils import get_logger

logger = get_logger("value_iteration")

MODEL_FILE = "model.npz"   # name of the model inside the source directory (Step 7 layout)


@dataclass
class SweepRecord:
    iteration: int
    max_delta: float           # max_s |V_{k+1}(s) - V_k(s)|
    mean_delta: float          # mean over the reported states
    states_updated: int        # reported states whose value changed by at least theta
    v_min: float               # V_{k+1} over the reported states
    v_mean: float
    v_max: float


@dataclass
class ValueIterationResult:
    converged: bool
    iterations: int
    bellman_residual: float    # max_s |(T V)(s) - V(s)| for the final V
    history: list[SweepRecord] = field(default_factory=list)
    seconds: float = 0.0


class ValueIteration:
    def __init__(self, mdp: TabularMDP, gamma: float, theta: float, max_iterations: int,
                 tie_tolerance: float = 1e-9, report_states: np.ndarray | None = None) -> None:
        if not 0 < gamma < 1:
            raise ValueError("gamma must be in (0, 1)")
        self.mdp = mdp
        self.gamma = gamma
        self.theta = theta
        self.max_iterations = max_iterations
        self.tie_tolerance = tie_tolerance
        # States summarised in the history (e.g. only modelled ones); every state is still updated.
        self.report_states = (np.ones(mdp.n_states, dtype=bool) if report_states is None
                              else np.asarray(report_states, dtype=bool))
        self.V = self.initialize_values()
        self.policy = greedy_actions(self.q_values(), tie_tolerance)
        self.result: ValueIterationResult | None = None

    @classmethod
    def from_config(cls, mdp: TabularMDP, cfg: ValueIterationConfig,
                    report_states: np.ndarray | None = None) -> "ValueIteration":
        return cls(mdp, cfg.gamma, cfg.theta, cfg.max_iterations, cfg.tie_tolerance, report_states)

    # ---------------------------------------------------------------- algorithm
    def initialize_values(self) -> np.ndarray:
        return np.zeros(self.mdp.n_states)

    def q_values(self, V: np.ndarray | None = None) -> np.ndarray:
        return q_values(self.mdp, self.V if V is None else V, self.gamma)

    def value_update(self, V: np.ndarray) -> np.ndarray:
        """One Bellman-optimality backup: (T V)(s) = max over admissible a of Q_V(s, a)."""
        return self.q_values(V).max(axis=1)

    def bellman_residual(self, V: np.ndarray | None = None) -> float:
        V = self.V if V is None else V
        return float(np.max(np.abs(self.value_update(V) - V)))

    def extract_policy(self, V: np.ndarray | None = None) -> np.ndarray:
        return greedy_actions(self.q_values(V), self.tie_tolerance)

    def train(self) -> ValueIterationResult:
        start = time.perf_counter()
        V = self.initialize_values()
        history: list[SweepRecord] = []
        converged = False
        rep = self.report_states
        for k in range(1, self.max_iterations + 1):
            V_new = self.value_update(V)
            diff = np.abs(V_new - V)
            max_delta = float(diff.max())
            history.append(SweepRecord(k, max_delta, float(diff[rep].mean()), int(np.sum(diff[rep] >= self.theta)),
                                       float(V_new[rep].min()), float(V_new[rep].mean()), float(V_new[rep].max())))
            V = V_new
            if max_delta < self.theta:
                converged = True
                break
        if not np.isfinite(V).all():
            raise FloatingPointError("value function is not finite")
        if not converged:
            logger.warning("did not reach theta=%g in %d iterations (delta %.2e)",
                           self.theta, self.max_iterations, history[-1].max_delta)
        self.V = V
        self.policy = self.extract_policy(V)
        self.result = ValueIterationResult(converged, len(history), self.bellman_residual(V), history,
                                           time.perf_counter() - start)
        logger.info("%s after %d iterations, final delta %.2e, Bellman residual %.2e",
                    "converged" if converged else "stopped", len(history), history[-1].max_delta,
                    self.result.bellman_residual)
        return self.result

    # ------------------------------------------------------------------ queries
    def get_policy(self) -> np.ndarray:
        return self.policy.copy()

    def get_value_function(self) -> np.ndarray:
        return self.V.copy()

    def predict(self, state: int) -> int:
        return int(self.policy[int(state)])


class ValueIterationController(PolicyIterationController):
    """Runs a Value Iteration policy in the tabular environment (same bookkeeping as the PI controller)."""

    name = "value_iteration"


# --------------------------------------------------------------------------- #
# Reusing the Step 7 model + persistence
# --------------------------------------------------------------------------- #
@dataclass
class SourceModel:
    """The saved empirical model and the Policy Iteration artifact it came from."""

    directory: Path
    model: EmpiricalModel
    model_hash: str
    source_payload: dict


def load_source_model(directory: Path) -> SourceModel:
    """Load the Step 7 empirical model without rebuilding it; verify its hash if one is recorded."""
    payload = json.loads((directory / POLICY_FILE).read_text(encoding="utf-8"))
    model = load_model(directory / payload["model"].get("file", MODEL_FILE), payload["model"])
    digest = model_hash(model)
    stored = payload["model"].get("model_hash")
    if stored is not None and stored != digest:
        raise ValueError(f"model in {directory} does not match its recorded model_hash")
    return SourceModel(directory, model, digest, payload)


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def save(vi: ValueIteration, source: SourceModel, config: Config, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    r = vi.result
    payload = {
        "algorithm": "value_iteration",
        "gamma": vi.gamma,
        "theta": vi.theta,
        "tie_tolerance": vi.tie_tolerance,
        "hyperparameters": config.rl.value_iteration.model_dump(),
        "model": {"source_dir": _relative(source.directory), "file": source.source_payload["model"].get("file", MODEL_FILE),
                  "model_hash": source.model_hash,
                  "modelled_states": np.flatnonzero(source.model.modelled).tolist(),
                  "note": "Step 7 empirical model reused as-is; Value Iteration performs no sampling"},
        **space_definitions(config),
        "initial_values": "V_0(s) = 0 for all states",
        "policy": vi.policy.tolist(),
        "value_function": vi.V.tolist(),
        "admissible": vi.mdp.admissible.astype(int).tolist(),
        "training": {"converged": r.converged, "iterations": r.iterations, "bellman_residual": r.bellman_residual,
                     "seconds": r.seconds, "history": [asdict(h) for h in r.history]} if r else None,
    }
    path = directory / POLICY_FILE
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return path


@dataclass
class LoadedValueIteration:
    policy: np.ndarray
    value_function: np.ndarray
    payload: dict
    source: SourceModel

    def predict(self, state: int) -> int:
        return int(self.policy[int(state)])


def load(directory: Path) -> LoadedValueIteration:
    payload = json.loads((directory / POLICY_FILE).read_text(encoding="utf-8"))
    source_dir = Path(payload["model"]["source_dir"])
    source = load_source_model(source_dir if source_dir.is_absolute() else PROJECT_ROOT / source_dir)
    if source.model_hash != payload["model"]["model_hash"]:
        raise ValueError(f"the model in {source.directory} is not the one this policy was trained on")
    return LoadedValueIteration(np.asarray(payload["policy"], dtype=np.int64),
                                np.asarray(payload["value_function"], dtype=np.float64), payload, source)


def train_value_iteration(config: Config | None = None, model_dir: Path | None = None
                          ) -> tuple[ValueIteration, SourceModel]:
    config = config or get_config()
    cfg = config.rl.value_iteration
    source = load_source_model(model_dir or cfg.model_path)
    source_gamma = source.source_payload.get("gamma")
    if source_gamma is not None and source_gamma != cfg.gamma:
        raise ValueError(f"value_iteration.gamma={cfg.gamma} differs from the gamma={source_gamma} of the "
                         f"Policy Iteration run that produced this model; the comparison would not be like-for-like")
    vi = ValueIteration.from_config(source.model.to_mdp(), cfg, report_states=source.model.modelled)
    vi.train()
    return vi, source


def main() -> None:
    cfg = get_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, default=cfg.rl.value_iteration.model_path)
    parser.add_argument("--out", type=Path, default=cfg.paths.models / "value_iteration")
    args = parser.parse_args()

    vi, source = train_value_iteration(cfg, args.model_dir)
    path = save(vi, source, cfg, args.out)
    r = vi.result
    print(f"Value Iteration {'converged' if r.converged else 'did NOT converge'} in {r.iterations} iterations "
          f"(Bellman residual {r.bellman_residual:.2e}, {r.seconds:.2f} s) on model {source.model_hash[:12]}... "
          f"from {source.directory}")
    logger.info("wrote %s", path)


if __name__ == "__main__":
    main()
