import os
import time
from typing import Any, Dict, Optional

import h5py
import numpy as np
import torch

from agent_factory.agents.base_agent import BaseAgent
from agent_factory.agents.registry import register_agent


@register_agent("Identity")
class IdentityAgent(BaseAgent):
    """
    Minimal inference-only agent for runner hardware tests.

    By default this agent emits a deterministic absolute-joint chunk for Piper
    hardware motion tests. If cfg.agent_sp.replay_h5_path is set, it instead
    replays root env-native actions from a rollout H5 file.
    """

    uses_env_safe_action = False
    CHUNK_HORIZON = 64
    BUMP_AMPLITUDE_RAD = 0.4
    BASE_ACTION = (
        0.400, 1.000, -0.900, 0.099, 0.901, -0.199, 0.050,
        -0.399, 1.000, -0.899, 0.100, 0.900, -0.199, 0.049,
    )

    def _init_components(self):
        self.action_dim = int(
            getattr(
                getattr(self.cfg, "actor", None),
                "action_dim",
                getattr(self.cfg.env, "action_dim", 1),
            )
        )
        self.pred_horizon = int(
            getattr(
                getattr(self.cfg, "actor", None),
                "pred_horizon",
                getattr(self.cfg.env, "pred_horizon", self.CHUNK_HORIZON),
            )
        )
        self.act_horizon = int(getattr(self.cfg.env, "act_horizon", self.pred_horizon))
        self._replay_cursor = 0
        self._replay_actions: Optional[torch.Tensor] = None
        self._replay_loop = False

        agent_sp = getattr(self.cfg, "agent_sp", None)
        self.sample_delay_sec = float(self._cfg_get(agent_sp, "sample_delay_sec", 0.0) or 0.0)
        if self.sample_delay_sec > 0.0:
            print(
                f"[IdentityAgent] sample_action delay enabled: "
                f"{self.sample_delay_sec * 1000.0:.1f}ms"
            )
        replay_h5_path = self._cfg_get(agent_sp, "replay_h5_path", "")
        if replay_h5_path:
            replay_traj_key = self._cfg_get(agent_sp, "replay_traj_key", "")
            replay_action_key = self._cfg_get(agent_sp, "replay_action_key", "action")
            self._replay_loop = bool(self._cfg_get(agent_sp, "replay_loop", False))
            replay_np = self._load_replay_actions(
                h5_path=str(replay_h5_path),
                traj_key=str(replay_traj_key or ""),
                action_key=str(replay_action_key or "action"),
            )
            self._replay_actions = torch.as_tensor(
                replay_np, dtype=torch.float32, device=self.device
            )
            print(
                "[IdentityAgent] Replay mode enabled: "
                f"{self._replay_actions.shape[0]} actions from {replay_h5_path}"
            )

    def _init_optimizers(self):
        return

    @staticmethod
    def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
        if cfg is None:
            return default
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        try:
            return getattr(cfg, key)
        except Exception:
            return default

    def _align_action_array(self, actions: np.ndarray, source: str) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2:
            raise ValueError(
                f"[IdentityAgent] Replay action dataset '{source}' must be 2-D, "
                f"got shape {actions.shape}."
            )
        if actions.shape[1] == self.action_dim:
            return actions
        if actions.shape[1] > self.action_dim:
            print(
                f"[IdentityAgent] Replay action dim {actions.shape[1]} > "
                f"env action_dim {self.action_dim}; cropping trailing dims."
            )
            return actions[:, : self.action_dim]
        print(
            f"[IdentityAgent] Replay action dim {actions.shape[1]} < "
            f"env action_dim {self.action_dim}; padding trailing zeros."
        )
        pad = np.zeros((actions.shape[0], self.action_dim - actions.shape[1]), dtype=np.float32)
        return np.concatenate([actions, pad], axis=1)

    def _load_replay_actions(self, h5_path: str, traj_key: str, action_key: str) -> np.ndarray:
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"[IdentityAgent] replay_h5_path not found: {h5_path}")

        chunks = []
        with h5py.File(h5_path, "r") as h5_file:
            if traj_key and traj_key.lower() != "all":
                if traj_key not in h5_file:
                    raise KeyError(f"[IdentityAgent] replay_traj_key not found: {traj_key}")
                node = h5_file[traj_key]
                if isinstance(node, h5py.Dataset):
                    actions = node[()]
                    source = traj_key
                else:
                    if action_key not in node:
                        raise KeyError(
                            f"[IdentityAgent] action key '{action_key}' not found in {traj_key}."
                        )
                    actions = node[action_key][()]
                    source = f"{traj_key}/{action_key}"
                chunks.append(self._align_action_array(actions, source))
            elif action_key in h5_file:
                chunks.append(self._align_action_array(h5_file[action_key][()], action_key))
            else:
                group_names = sorted(
                    key
                    for key in h5_file.keys()
                    if key.startswith("traj_")
                    and isinstance(h5_file[key], h5py.Group)
                    and action_key in h5_file[key]
                )
                if not group_names:
                    raise KeyError(
                        f"[IdentityAgent] No root '{action_key}' dataset or "
                        f"'traj_*/{action_key}' datasets found in {h5_path}."
                    )
                for group_name in group_names:
                    source = f"{group_name}/{action_key}"
                    chunks.append(
                        self._align_action_array(h5_file[group_name][action_key][()], source)
                    )

        actions = np.concatenate(chunks, axis=0)
        if actions.shape[0] == 0:
            raise ValueError(f"[IdentityAgent] Replay action sequence is empty: {h5_path}")
        return actions

    def _base_action_tensor(self) -> torch.Tensor:
        base = torch.tensor(self.BASE_ACTION, dtype=torch.float32, device=self.device)
        if self.action_dim == base.numel():
            return base
        if self.action_dim < base.numel():
            return base[:self.action_dim]
        pad = torch.zeros(self.action_dim - base.numel(), dtype=torch.float32, device=self.device)
        return torch.cat([base, pad], dim=0)

    def _replay_action_chunk(self) -> torch.Tensor:
        assert self._replay_actions is not None
        num_actions = int(self._replay_actions.shape[0])
        indices = torch.arange(
            self._replay_cursor,
            self._replay_cursor + self.pred_horizon,
            device=self.device,
        )
        if self._replay_loop:
            indices = indices.remainder(num_actions)
        else:
            indices = torch.clamp(indices, max=num_actions - 1)

        chunk = self._replay_actions.index_select(0, indices.long())
        self._replay_cursor += max(1, self.act_horizon)
        if self._replay_loop:
            self._replay_cursor %= num_actions
        else:
            self._replay_cursor = min(self._replay_cursor, num_actions - 1)
        return chunk

    def sample_action(self, obs: Dict[str, Any], initial_noise: Optional[torch.Tensor] = None):
        """
        Return a [B, pred_horizon, action_dim] env-native action chunk.

        In replay mode the chunk comes from cfg.agent_sp.replay_h5_path and the
        internal cursor advances by cfg.env.act_horizon, matching runner
        consumption. In deterministic mode, dimensions are 1-indexed in the
        experiment note:
        - dim 4  -> index 3
        - dim 11 -> index 10
        """
        del initial_noise
        if self.sample_delay_sec > 0.0:
            time.sleep(self.sample_delay_sec)

        batch_size = 1
        if isinstance(obs, dict):
            for value in obs.values():
                if isinstance(value, torch.Tensor) and value.ndim > 0:
                    batch_size = int(value.shape[0])
                    break

        if self._replay_actions is not None:
            chunk = self._replay_action_chunk()
        else:
            base = self._base_action_tensor()
            chunk = base.unsqueeze(0).repeat(self.pred_horizon, 1)
            bump = torch.sin(
                torch.linspace(0.0, torch.pi, self.pred_horizon, device=self.device)
            ) * self.BUMP_AMPLITUDE_RAD
            if self.action_dim > 3:
                chunk[:, 3] = base[3] + bump
            if self.action_dim > 10:
                chunk[:, 10] = base[10] + bump
        return chunk.unsqueeze(0).repeat(batch_size, 1, 1)

    def load(self, path: str):
        print(f"[IdentityAgent] Skip loading checkpoint for deterministic motion test agent: {path}")
        return {}

    def save(self, path: str, meta: Dict = None):
        print(f"[IdentityAgent] No parameters to save for deterministic motion test agent: {path}")

    def start_train(self, dataset, additional_args: Optional[Dict[str, Any]] = None):
        raise NotImplementedError("IdentityAgent is inference-only and cannot be trained.")
