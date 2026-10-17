"""Uniform experience replay for DQN (Step 10).

Each transition stores the action masks observed *at the time* (current and next state), so a sampled target
never depends on a simulator state that has since changed. Storage is a fixed-capacity ring buffer of NumPy
arrays; the oldest transition is overwritten once the buffer is full. Sampling is uniform with replacement and
uses a generator passed in by the caller, so it is reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Batch:
    states: np.ndarray        # float32 (B, obs_dim)
    actions: np.ndarray       # int64   (B,)
    rewards: np.ndarray       # float32 (B,)
    next_states: np.ndarray   # float32 (B, obs_dim)
    terminated: np.ndarray    # bool    (B,)
    truncated: np.ndarray     # bool    (B,)
    masks: np.ndarray         # bool    (B, n_actions)  valid actions in `states`
    next_masks: np.ndarray    # bool    (B, n_actions)  valid actions in `next_states`


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, n_actions: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity, self.obs_dim, self.n_actions = capacity, obs_dim, n_actions
        self.states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=bool)
        self.truncated = np.zeros(capacity, dtype=bool)
        self.masks = np.zeros((capacity, n_actions), dtype=bool)
        self.next_masks = np.zeros((capacity, n_actions), dtype=bool)
        self.position = 0     # next slot to write
        self.size = 0
        self.total_added = 0

    def __len__(self) -> int:
        return self.size

    def add(self, state, action: int, reward: float, next_state, terminated: bool, truncated: bool,
            mask, next_mask) -> None:
        state, next_state = np.asarray(state, dtype=np.float32), np.asarray(next_state, dtype=np.float32)
        if state.shape != (self.obs_dim,) or next_state.shape != (self.obs_dim,):
            raise ValueError(f"states must have shape ({self.obs_dim},)")
        if not 0 <= action < self.n_actions:
            raise ValueError(f"action {action} outside [0, {self.n_actions})")
        i = self.position
        self.states[i], self.actions[i], self.rewards[i] = state, action, reward
        self.next_states[i], self.terminated[i], self.truncated[i] = next_state, terminated, truncated
        self.masks[i] = np.asarray(mask, dtype=bool)
        self.next_masks[i] = np.asarray(next_mask, dtype=bool)
        self.position = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.total_added += 1

    def sample(self, batch_size: int, rng: np.random.Generator) -> Batch:
        if batch_size > self.size:
            raise ValueError(f"cannot sample {batch_size} from {self.size} stored transitions")
        idx = rng.integers(0, self.size, size=batch_size)
        return Batch(self.states[idx], self.actions[idx], self.rewards[idx], self.next_states[idx],
                     self.terminated[idx], self.truncated[idx], self.masks[idx], self.next_masks[idx])
