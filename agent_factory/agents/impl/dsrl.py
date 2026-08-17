from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch

from agent_factory.agents.base_agent import BaseAgent
from agent_factory.agents.mixins.actor.dsrl_actor import DSRLActorMixin
from agent_factory.agents.mixins.critic.dsrl_critic import DSRLCriticMixin
from agent_factory.agents.registry import register_agent
from agent_factory.config.manager import ConfigManager
from agent_factory.config.structure import DiffusionBasePolicyConfig


@dataclass
class AgentSpecialConfig:
    iters: int = 60000
    log_interval: int = 100
    save_dir: str = "run_results"
    exp_name: str = ""
    base_policy: DiffusionBasePolicyConfig = field(default_factory=DiffusionBasePolicyConfig)
    pretrain_diffusion: bool = False
    pretrain_iters: int = 50000


class MainMixin:
    CONFIG_CLASS = AgentSpecialConfig
    CONFIG_KEY = "agent_sp"


@register_agent("dsrl")
class DSRLAgent(MainMixin, DSRLActorMixin, DSRLCriticMixin, BaseAgent):
    """
    DSRL (Diffusion Steering Reinforcement Learning) Agent.
    """
    def _init_components(self):
        self._build_critic()
        self._build_actor()

    def _init_optimizers(self):
        # Optimizers are already initialized in Mixins' _build_ methods
        pass

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        # 1. Update Critics
        critic_metrics = self.update_critic(batch)
        
        # 2. Update Actor
        actor_metrics = self.update_actor(batch)
        
        # 3. Update Temperature
        temp_metrics = self.update_temperature(actor_metrics["entropy_log_prob_mean"])
        
        # 4. Target Update
        self.soft_update_target()
        
        self.step += 1
        return {
            **critic_metrics,
            **actor_metrics,
            **temp_metrics
        }

    def train_loop(self, dataloader, num_steps: int, save_dir: str = ""):
        from tqdm import tqdm
        import gc

        self.train()
        log_interval = max(int(self.cfg.agent_sp.log_interval), 1)
        save_interval = max(int(getattr(self.cfg.train, "save_interval", num_steps // 4)), 1)

        def infinite_iterator(loader):
            while True:
                for batch in loader:
                    yield batch

        it_obj = infinite_iterator(dataloader)
        
        running = {
            "loss_action_critic": 0.0,
            "loss_noise_critic": 0.0,
            "loss_actor": 0.0,
            "loss_temperature": 0.0,
        }
        pbar = tqdm(range(num_steps), desc="Train DSRL", leave=True)

        try:
            for step_idx in pbar:
                batch = self._batch_to_device(next(it_obj))
                metrics = self.update(batch)
                for key in running:
                    if key in metrics:
                        running[key] += float(metrics[key])

                if (step_idx + 1) % log_interval == 0:
                    pbar.set_postfix(
                        {
                            "a_critic": running["loss_action_critic"] / log_interval,
                            "n_critic": running["loss_noise_critic"] / log_interval,
                            "actor": running["loss_actor"] / log_interval,
                            "temp": running["loss_temperature"] / log_interval,
                        }
                    )
                    for key in running:
                        running[key] = 0.0

                if save_dir and (step_idx + 1) % save_interval == 0:
                    current_step = step_idx + 1
                    ckpt_path = os.path.join(save_dir, f"step_{current_step}.pth")
                    self.save(ckpt_path, meta={"step": current_step, "mode": "dsrl"})
        finally:
            del it_obj
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def start_train(self, dataset: Any, additional_args: Optional[Dict[str, Any]] = None):
        from torch.utils.data import DataLoader
        import os

        additional_args = additional_args or {}

        # Validate base policy reference before any heavy operation.
        self._validate_base_policy_reference(require_ready=False)

        # Optional pre-train logic: auto-generate missing base policy.
        cfg_sp = self.cfg.agent_sp
        if cfg_sp.pretrain_diffusion:
            base_cfg = cfg_sp.base_policy
            checkpoint_path = getattr(base_cfg, "ckpt_path", "").strip()
            
            # 如果没有路径或路径不存在，则触发预训练
            if not checkpoint_path or not os.path.exists(checkpoint_path):
                print(f"[DSRL] Base policy not found at '{checkpoint_path}'. Starting Pre-training...")
                self._pretrain_diffusion(dataset, additional_args)
                print("[DSRL] Base policy pre-training finished. Continue DSRL training in current run.")
            else:
                print(f"[DSRL] Found existing base policy at '{checkpoint_path}'. Skipping Pre-training.")

        # Strong validation before actual DSRL training.
        self._validate_base_policy_reference(require_ready=True)

        # Ensure base policy is loaded
        self.ensure_base_policy()

        # Resolve dataset
        if isinstance(dataset, dict):
            phase = additional_args.get("phase", "offline")
            train_dataset = dataset.get(phase) or dataset.get("offline") or dataset.get("online")
        else:
            train_dataset = dataset

        if train_dataset is None:
            raise ValueError("DSRL.start_train requires a valid dataset.")

        # Switch mode if supported (e.g., 'all' or 'success')
        if hasattr(train_dataset, "switch"):
            train_dataset.switch(additional_args.get("mode", "all"))

        # 统一拟合动作归一化器（DSRLCriticMixin 的环境动作归一化）
        self._fit_action_normalizer_from_dataset(train_dataset)

        cfg_sp = self.cfg.agent_sp
        if not cfg_sp.exp_name:
            cfg_sp.exp_name = f"dsrl_{self.cfg.env.env_id}"
        
        checkpoint_dir = os.path.join(cfg_sp.save_dir, cfg_sp.exp_name)
        os.makedirs(checkpoint_dir, exist_ok=True)

        loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.cfg.train.num_workers,
            pin_memory=True,
            persistent_workers=(self.cfg.train.num_workers > 0),
            prefetch_factor=2 if self.cfg.train.num_workers > 0 else None
        )

        num_steps = int(additional_args.get("num_steps", cfg_sp.iters))
        print(f"[DSRL] Starting training for {num_steps} steps...")
        self.train_loop(loader, num_steps, save_dir=checkpoint_dir)

        final_path = os.path.join(checkpoint_dir, f"{cfg_sp.exp_name}_final.pth")
        self.save(final_path, meta={"phase": "done"})

    def load(self, path: str):
        super().load(path)
        if self.base_policy is not None:
             self.base_policy.load()
             
    def _preprocess_obs(self, obs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        from agent_factory.data.utils import preprocess_obs
        return preprocess_obs(obs, device=self.device)

    def _pretrain_diffusion(self, dataset: Any, additional_args: Optional[Dict[str, Any]] = None):
        """
        内部执行 Diffusion Policy 的预训练 (BC)
        """
        from agent_factory.agents.registry import make_agent, get_default_config
        from agent_factory.config.resolution import general_resolve
        import os
        from omegaconf import OmegaConf

        # 1. 构造 Base Agent 的配置
        base_agent_type = "Diffusion_Vanilla"
        print(f"[DSRL Pre-train] Initializing {base_agent_type}...")
        
        # 获取默认配置并进行深度拷贝或 Merge 覆盖
        base_cfg = get_default_config(base_agent_type)
        
        # 复用当前 DSRL 的环境、数据集、设备和基础 Actor 参数
        # 注意：Base Agent 需要环境的 action_dim
        base_cfg.device = str(self.device)
        base_cfg.env = self.cfg.env
        base_cfg.dataset = self.cfg.dataset
        base_cfg.train = self.cfg.train
        
        # 针对 Actor 的对齐
        base_cfg.actor.obs_horizon = self.cfg.actor.obs_horizon
        base_cfg.actor.action_dim = self.cfg.env.action_dim
        base_cfg.actor.pred_horizon = self.cfg.env.pred_horizon
        base_cfg.actor.encoder = self.cfg.actor.encoder
        
        # 针对 Agent_SP 的对齐
        base_cfg.agent_sp.save_dir = self.cfg.agent_sp.save_dir
        base_cfg.agent_sp.exp_name = f"{self.cfg.agent_sp.exp_name or 'dsrl'}_pretrain"
        base_cfg.train.actor_iters = self.cfg.agent_sp.pretrain_iters

        base_cfg, _runtime_spec = general_resolve(file_config=base_cfg)
        base_agent = make_agent(base_agent_type, base_cfg)
        
        # 2. 开始预训练
        print(f"[DSRL Pre-train] Starting BC training for {base_cfg.train.actor_iters} steps...")
        

        # 3. 记录产出的权重路径并回填到当前配置
        pretrain_dir = os.path.join(base_cfg.agent_sp.save_dir, base_cfg.agent_sp.exp_name)
        final_ckpt = os.path.join(pretrain_dir, f"{base_cfg.agent_sp.exp_name}_final.pth")
        
        ConfigManager.save_config(base_cfg, save_dir=pretrain_dir, filename= 'base_config.yaml')
        base_agent.start_train(dataset, additional_args)

        if os.path.exists(final_ckpt):
            print(f"[DSRL Pre-train] Pre-training finished. Final checkpoint: {final_ckpt}")
            self.cfg.agent_sp.base_policy.ckpt_path = final_ckpt
            self.cfg.agent_sp.base_policy.type = base_agent_type
            self.cfg.agent_sp.base_policy.base_config_path = os.path.join(pretrain_dir, "base_config.yaml")


        else:
            # 兜底：尝试找 step_ 结尾的最新一个
            print(f"[DSRL Pre-train] Warning: Final checkpoint not found at {final_ckpt}. DSRL might fail to load it.")
            # 我们不抛异常，让 self.ensure_base_policy() 在 load 时报错，这样用户能看到更直接的错误

    def _validate_base_policy_reference(self, require_ready: bool):
        """
        Validate DSRL base policy metadata under agent_sp.base_policy.
        When require_ready=True, ckpt/config files must exist and match key env dimensions.
        """
        from omegaconf import OmegaConf

        bp = getattr(self.cfg.agent_sp, "base_policy", None)
        if bp is None:
            raise ValueError("DSRL requires 'agent_sp.base_policy' configuration.")

        policy_type = getattr(bp, "type", "").strip()
        ckpt_path = getattr(bp, "ckpt_path", "").strip()
        base_config_path = getattr(bp, "base_config_path", "").strip()

        missing = []
        if not policy_type:
            missing.append("type")
        if not ckpt_path:
            missing.append("ckpt_path")
        if not base_config_path:
            missing.append("base_config_path")

        if missing and require_ready:
            raise ValueError(
                "DSRL base_policy is incomplete. Missing fields under agent_sp.base_policy: "
                + ", ".join(missing)
            )

        if not require_ready:
            return

        if not os.path.exists(ckpt_path):
            raise ValueError(f"DSRL base_policy ckpt not found: {ckpt_path}")
        if not os.path.exists(base_config_path):
            raise ValueError(f"DSRL base_policy config not found: {base_config_path}")

        base_cfg = OmegaConf.load(base_config_path)
        if not hasattr(base_cfg, "env"):
            raise ValueError(f"Invalid base policy config (missing env): {base_config_path}")

        env = self.cfg.env
        base_env = base_cfg.env
        checks = {
            "action_dim": (int(getattr(base_env, "action_dim", -1)), int(env.action_dim)),
            "pred_horizon": (int(getattr(base_env, "pred_horizon", -1)), int(env.pred_horizon)),
            "obs_horizon": (int(getattr(base_env, "obs_horizon", -1)), int(env.obs_horizon)),
        }
        mismatches = [
            f"{k}: base={v[0]} vs dsrl={v[1]}"
            for k, v in checks.items()
            if v[0] != v[1]
        ]
        if mismatches:
            raise ValueError(
                "Base policy config mismatch with DSRL env: " + "; ".join(mismatches)
            )
