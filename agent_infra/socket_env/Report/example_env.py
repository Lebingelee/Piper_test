"""A dependency-light, standard Gymnasium environment example for reporters."""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CounterEnv(gym.Env):
    observation_space = spaces.Dict({"state": spaces.Dict({"counter": spaces.Box(-1_000_000, 1_000_000, shape=(1,), dtype=np.float32)})})
    action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
    metadata = {"meta_keys": {"obs": {"state": {"counter": [1]}}, "action": {}}, "observation_order": {"state": ["counter"]}}

    def __init__(self, max_steps: int = 3):
        self.max_steps, self._count = int(max_steps), 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._count = 0
        return {"state": {"counter": np.array([0.0], dtype=np.float32)}}, {"reset": True}

    def step(self, action):
        self._count += 1
        return {"state": {"counter": np.array([self._count], dtype=np.float32)}}, float(np.asarray(action).sum()), False, self._count >= self.max_steps, {"count": self._count}


def make_env(**kwargs):
    return CounterEnv(**kwargs)
