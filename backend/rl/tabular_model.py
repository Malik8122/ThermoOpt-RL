"""Empirical model of the 144-state aggregated MDP, for Policy Iteration (and Value Iteration in Step 8).

The tabular environment has no closed-form transition function: the abstract state hides rack
identities, job queues, clock time and the cooling ramp. The model is therefore *estimated* from the
real environment (certainty-equivalence planning, PHASE1_DESIGN §4.4):

1. **Collect concrete states.** Run ``collection_episodes`` tabular episodes of ``collection_minutes``
   with a uniform-random *valid* policy (``env.action_masks()``) and random start hours, so both carbon
   levels and the whole daily cycle are covered. Every ``snapshot_interval_minutes`` a deep copy of the
   simulator is stored under its abstract state, reservoir-sampled to at most ``max_snapshots_per_state``.
2. **Sample transitions.** For each stored abstract state s and each of the 9 actions a, take
   ``samples_per_state_action`` one-step transitions. Sample k starts from stored snapshot k mod n_s,
   reseeds the simulator's random streams (arrivals, thermal noise) with ``derive_seed(seed, s, a, k)``,
   and calls the unmodified ``DataCenterEnv.step``. That call resolves the macro-action, validates it
   (an invalid action is a no-op with the environment's invalid-action penalty), advances the twin and
   scores the reward. The model records s', r and whether the action was valid.
3. **Closure.** A successor s' with no stored snapshot gets the concrete successor state as its
   snapshot and is modelled in turn, so every state the model can reach has a model.

Estimates: ``P(s'|s,a) = N(s,a,s') / N(s,a)``, ``R(s,a,s') = mean reward over those transitions``.

The environment never terminates (it is a continuing task truncated only by a time limit), so the model
has no terminal states and each one-step sample starts with a fresh step counter.

Limitation: the abstract state is not Markov. P and R are averages over whichever concrete states the
collection policy happened to visit, and they depend on that distribution.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field

import numpy as np

from backend.config import Config, get_config
from backend.rl.actions import CoolingCommand, TabularPlacement, encode_tabular_action
from backend.rl.environment import DataCenterEnv, EpisodeStats
from backend.simulation.datacentre import DataCentre
from backend.utils.logging_utils import get_logger
from backend.utils.seeding import derive_seed, make_rng

logger = get_logger("tabular_model")

# Delay the head job (a no-op on an empty queue) + maintain cooling: valid in every concrete state.
ALWAYS_VALID_ACTION = encode_tabular_action(TabularPlacement.DELAY, CoolingCommand.MAINTAIN)


@dataclass
class TabularMDP:
    """A finite MDP with an admissible-action mask.

    P[s, a, s']    transition probabilities; each admissible row sums to 1
    R[s, a, s']    expected reward of that transition (0 where P = 0)
    admissible     bool[s, a]; the actions a policy may choose in s
    """

    P: np.ndarray
    R: np.ndarray
    admissible: np.ndarray

    def __post_init__(self) -> None:
        self.P = np.asarray(self.P, dtype=np.float64)
        self.R = np.asarray(self.R, dtype=np.float64)
        self.admissible = np.asarray(self.admissible, dtype=bool)
        S, A, S2 = self.P.shape
        if S != S2 or self.R.shape != self.P.shape or self.admissible.shape != (S, A):
            raise ValueError("P and R must be (S, A, S) and admissible (S, A)")
        if not self.admissible.any(axis=1).all():
            raise ValueError("every state needs at least one admissible action")
        if not (np.isfinite(self.P).all() and np.isfinite(self.R).all()) or (self.P < 0).any():
            raise ValueError("P must be finite and non-negative, R finite")
        sums = self.P.sum(axis=2)
        if not np.allclose(sums[self.admissible], 1.0, atol=1e-9):
            raise ValueError("transition probabilities of admissible actions must sum to 1")

    @property
    def n_states(self) -> int:
        return self.P.shape[0]

    @property
    def n_actions(self) -> int:
        return self.P.shape[1]

    @property
    def expected_reward(self) -> np.ndarray:
        """R(s, a) = sum_s' P(s'|s,a) R(s,a,s')."""
        return np.einsum("sat,sat->sa", self.P, self.R)


@dataclass
class EmpiricalModel:
    """Raw sample statistics plus the MDP derived from them."""

    counts: np.ndarray              # int[s, a, s']
    reward_sum: np.ndarray          # float[s, a, s']
    valid_count: np.ndarray         # int[s, a]
    snapshots_per_state: np.ndarray  # int[s], concrete states sampled from (0 = unmodelled)
    metadata: dict = field(default_factory=dict)

    @property
    def n_samples(self) -> np.ndarray:
        return self.counts.sum(axis=2)

    @property
    def modelled(self) -> np.ndarray:
        """States with a sampled model. Other states were never reached; see ``to_mdp``."""
        return self.snapshots_per_state > 0

    @property
    def valid_fraction(self) -> np.ndarray:
        n = self.n_samples
        return np.divide(self.valid_count, n, out=np.zeros(n.shape), where=n > 0)

    def to_mdp(self) -> TabularMDP:
        """Maximum-likelihood MDP.

        Admissible actions: in a modelled state, every action that was valid in at least one sample.
        Validity depends on the concrete state (cooling at 80 % vs 100 % are both HIGH; whether the head
        job fits depends on the racks), so an admissible action may still be invalid in some concrete
        states. Its estimated P and R include those invalid outcomes exactly as the environment produces
        them (no-op + invalid-action penalty), so every action is scored consistently.

        Unmodelled states (never reached by collection or by any sampled transition) get an absorbing
        self-loop with reward 0 and only the always-valid action. By the closure step no modelled state
        can transition into them, so they cannot affect any modelled value.
        """
        S, A, _ = self.counts.shape
        n = self.n_samples
        P = np.zeros((S, A, S))
        R = np.zeros((S, A, S))
        np.divide(self.counts, n[..., None], out=P, where=n[..., None] > 0)
        np.divide(self.reward_sum, self.counts, out=R, where=self.counts > 0)
        admissible = self.modelled[:, None] & (self.valid_count > 0)
        for s in np.flatnonzero(~self.modelled):
            P[s, :, s] = 1.0
            admissible[s, ALWAYS_VALID_ACTION] = True
        return TabularMDP(P, R, admissible)


# --------------------------------------------------------------------------- #
# Model construction
# --------------------------------------------------------------------------- #
def reseed_simulator(dc: DataCentre, seed: int) -> None:
    """Give a copied simulator fresh, independent arrival and thermal-noise streams.

    Without this every copy of a snapshot would replay the same random future. The simulator is not
    modified; only its two random generators are replaced, exactly as ``DataCentre.__init__`` builds them.
    """
    dc._thermal_rng = make_rng(derive_seed(seed, "thermal"))
    dc.generator.rng = make_rng(derive_seed(seed, "workloads"))


def collect_snapshots(config: Config, seed: int) -> dict[int, list[DataCentre]]:
    """Step 1: concrete simulator states grouped by abstract state."""
    mcfg = config.rl.tabular_model
    env = DataCenterEnv(representation="tabular", episode_minutes=mcfg.collection_minutes,
                        randomize_start_hour=True, config=config)
    rng = make_rng(derive_seed(seed, "tabular-model", "collection"))
    pool: dict[int, list[DataCentre]] = {}
    seen: dict[int, int] = {}
    for episode in range(mcfg.collection_episodes):
        state, _ = env.reset(seed=derive_seed(seed, "tabular-model", "episode", episode))
        truncated = False
        while not truncated:
            if env.dc.minute % mcfg.snapshot_interval_minutes == 0:
                _reservoir_add(pool, seen, int(state), env.dc, mcfg.max_snapshots_per_state, rng)
            action = int(rng.choice(np.flatnonzero(env.action_masks())))
            state, _, _, truncated, _ = env.step(action)
    return pool


def _reservoir_add(pool, seen, state: int, dc: DataCentre, capacity: int, rng: np.random.Generator) -> None:
    seen[state] = seen.get(state, 0) + 1
    bucket = pool.setdefault(state, [])
    if len(bucket) < capacity:
        bucket.append(copy.deepcopy(dc))
    else:
        j = int(rng.integers(seen[state]))
        if j < capacity:
            bucket[j] = copy.deepcopy(dc)


def build_empirical_model(config: Config | None = None, seed: int | None = None,
                          samples_per_state_action: int | None = None) -> EmpiricalModel:
    """Steps 1-3 of the module docstring. Deterministic for a given (config, seed, sample count)."""
    config = config or get_config()
    seed = config.project.seed if seed is None else seed
    n_samples = samples_per_state_action or config.rl.policy_iteration.model_samples_per_state_action
    mcfg = config.rl.tabular_model

    pool = collect_snapshots(config, seed)
    collected_states = sorted(pool)
    logger.info("collected snapshots for %d abstract states", len(collected_states))

    env = DataCenterEnv(representation="tabular", episode_minutes=config.simulation.episode_minutes, config=config)
    env.reset(seed=seed)
    S, A = env.observation_space.n, env.action_space.n
    counts = np.zeros((S, A, S), dtype=np.int64)
    reward_sum = np.zeros((S, A, S))
    valid_count = np.zeros((S, A), dtype=np.int64)
    snapshots = np.zeros(S, dtype=np.int64)

    todo = list(collected_states)
    processed: set[int] = set()
    discovered: list[int] = []
    while todo:
        s = todo.pop(0)
        bucket = pool[s]
        snapshots[s] = len(bucket)
        processed.add(s)
        for a in range(A):
            for k in range(n_samples):
                env.dc = copy.deepcopy(bucket[k % len(bucket)])
                reseed_simulator(env.dc, derive_seed(seed, "tabular-model", "sample", s, a, k))
                env.stats = EpisodeStats()
                s_next, reward, terminated, _, info = env.step(a)
                s_next = int(s_next)
                if terminated:  # the environment is continuing; a terminal transition would need V(s') = 0
                    raise RuntimeError("unexpected terminal transition in a continuing environment")
                counts[s, a, s_next] += 1
                reward_sum[s, a, s_next] += reward
                valid_count[s, a] += int(info["action_valid"])
                if s_next not in processed:
                    if s_next not in pool:
                        pool[s_next] = []
                        todo.append(s_next)
                        discovered.append(s_next)
                    if len(pool[s_next]) < mcfg.max_snapshots_per_state and s_next in todo:
                        pool[s_next].append(copy.deepcopy(env.dc))
        logger.debug("modelled state %d from %d snapshots", s, len(bucket))

    logger.info("modelled %d states (%d found by closure); %d x %d samples each",
                int((snapshots > 0).sum()), len(discovered), A, n_samples)
    metadata = {
        "seed": seed,
        "samples_per_state_action": n_samples,
        "collection": mcfg.model_dump(),
        "collection_policy": "uniform over valid tabular actions, random start hour",
        "states_from_collection": collected_states,
        "states_from_closure": discovered,
    }
    return EmpiricalModel(counts, reward_sum, valid_count, snapshots, metadata)


# --------------------------------------------------------------------------- #
# Shared dynamic-programming helpers (Policy Iteration, Value Iteration)
# --------------------------------------------------------------------------- #
def q_values(mdp: TabularMDP, V: np.ndarray, gamma: float) -> np.ndarray:
    """Q(s,a) = sum_s' P(s'|s,a) [R(s,a,s') + gamma V(s')]; -inf for inadmissible actions."""
    Q = mdp.expected_reward + gamma * (mdp.P @ V)
    return np.where(mdp.admissible, Q, -np.inf)


def tied_best(Q: np.ndarray, tie_tolerance: float) -> np.ndarray:
    """bool[s, a]: actions whose Q is within ``tie_tolerance`` of the state's maximum (never -inf ones)."""
    return Q >= Q.max(axis=1, keepdims=True) - tie_tolerance


def greedy_actions(Q: np.ndarray, tie_tolerance: float) -> np.ndarray:
    """Greedy policy; ties (within ``tie_tolerance``) go to the lowest action index."""
    return np.argmax(tied_best(Q, tie_tolerance), axis=1).astype(np.int64)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
MODEL_ARRAYS = ("counts", "reward_sum", "valid_count", "snapshots_per_state")


def model_hash(model: EmpiricalModel) -> str:
    """SHA-256 of the raw sample statistics (name, dtype, shape and bytes of each array).

    Two planners that report the same hash were given exactly the same empirical model.
    """
    h = hashlib.sha256()
    for name in MODEL_ARRAYS:
        arr = np.ascontiguousarray(getattr(model, name))
        h.update(f"{name}|{arr.dtype.str}|{arr.shape}|".encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def save_model(model: EmpiricalModel, path) -> None:
    np.savez_compressed(path, counts=model.counts, reward_sum=model.reward_sum,
                        valid_count=model.valid_count, snapshots_per_state=model.snapshots_per_state)


def load_model(path, metadata: dict | None = None) -> EmpiricalModel:
    with np.load(path) as data:
        return EmpiricalModel(data["counts"], data["reward_sum"], data["valid_count"],
                              data["snapshots_per_state"], metadata or {})

