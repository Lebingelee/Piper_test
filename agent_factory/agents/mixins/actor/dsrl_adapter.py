from __future__ import annotations

import os
from typing import Any, Dict, Optional, Union

import torch
from omegaconf import OmegaConf

from agent_factory.agents.registry import make_agent


class GroupRLDiffusionPolicyAdapter:
    def __init__(
        self,
        cfg,
        device: Union[torch.device, str],
        expected_action_dim: Optional[int] = None,
        expected_pred_horizon: Optional[int] = None,
        expected_obs_horizon: Optional[int] = None,
    ):
        self.cfg = cfg
        self.device = torch.device(device)
        self.agent = None
        self.expected_action_dim = expected_action_dim
        self.expected_pred_horizon = expected_pred_horizon
        self.expected_obs_horizon = expected_obs_horizon

    def load(self):
        if self.agent is not None:
            return self.agent

        checkpoint_path = getattr(self.cfg, "ckpt_path", "").strip()
        if not checkpoint_path:
            raise ValueError("DSRL base policy requires agent_sp.base_policy.ckpt_path.")

        base_config_path = getattr(self.cfg, "base_config_path", "").strip()
        if not base_config_path:
            raise ValueError("DSRL base policy requires agent_sp.base_policy.base_config_path.")
        if not os.path.exists(base_config_path):
            raise ValueError(f"DSRL base_policy.base_config_path not found: {base_config_path}")
        agent_cfg = OmegaConf.load(base_config_path)

        if "device" in agent_cfg:
            agent_cfg.device = str(self.device)

        agent_type = getattr(self.cfg, "type", "").strip()
        if not agent_type:
            raise ValueError("DSRL base policy requires agent_sp.base_policy.type.")
        
        # Registry should work here
        self.agent = make_agent(agent_type, agent_cfg)
        self.agent.load(checkpoint_path)
        self.agent.to(self.device)
        self.agent.eval()

        actor = getattr(self.agent, "actor", None)
        if actor is None or not hasattr(actor, "sample_action"):
            raise ValueError(
                "Loaded base policy checkpoint is not a valid agent with actor.sample_action()."
            )

        loaded_cfg = self.agent.cfg
        # Dimension checks
        if self.expected_action_dim is not None and int(getattr(loaded_cfg.env, "action_dim", -1)) != int(self.expected_action_dim):
            raise ValueError(f"Base policy action_dim mismatch: expected {self.expected_action_dim}, got {loaded_cfg.env.action_dim}.")
        if self.expected_pred_horizon is not None and int(getattr(loaded_cfg.env, "pred_horizon", -1)) != int(self.expected_pred_horizon):
            raise ValueError(f"Base policy pred_horizon mismatch: expected {self.expected_pred_horizon}, got {loaded_cfg.env.pred_horizon}.")
        if self.expected_obs_horizon is not None and int(getattr(loaded_cfg.env, "obs_horizon", -1)) != int(self.expected_obs_horizon):
             # Some models might have different obs_horizon but still work if we handle it, but for now strict check.
            raise ValueError(f"Base policy obs_horizon mismatch: expected {self.expected_obs_horizon}, got {loaded_cfg.env.obs_horizon}.")
            
        return self.agent

    def _to_device(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(self.device)
        if isinstance(value, dict):
            return {k: self._to_device(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._to_device(v) for v in value]
        return value

    def infer(self, obs: Dict[str, torch.Tensor], noise) -> torch.Tensor:
        agent = self.load()
        obs = self._to_device(obs)
        noise = torch.as_tensor(noise, device=self.device, dtype=torch.float32)

        # Base Agent should have its own _preprocess_obs
        processed_obs = agent._preprocess_obs(obs) if hasattr(agent, "_preprocess_obs") else obs

        # Ensure base policy stays in pure inference mode during DSRL training.
        was_training = bool(getattr(agent, "training", False))
        if hasattr(agent, "eval"):
            agent.eval()
        try:
            with torch.inference_mode():
                # Prefer base agent's unified inference API so all special logic
                # is owned by base policy config/implementation.
                if hasattr(agent, "sample_action"):
                    try:
                        action = agent.sample_action(processed_obs, initial_noise=noise)
                    except TypeError:
                        # Backward compatibility for non-diffusion or legacy sample_action signatures.
                        action = agent.sample_action(processed_obs)
                else:
                    raise NotImplementedError("Loaded base policy agent does not expose sample_action().")
        finally:
            if hasattr(agent, "train"):
                agent.train(was_training)

        return action
