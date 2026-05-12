import argparse
import copy
import json
import os
import sys
from typing import Any, Dict

import h5py
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent_factory.agents.impl.diffusion_cpiql_dac import DiffusionCPIQLDACAgent
from agent_factory.agents.registry import get_default_config, make_agent
from agent_factory.config.manager import ConfigManager
from agent_factory.data.impl.cpiql.expert_dataset import CPIQLExpertDataset


def _read_json_dataset(dataset) -> Dict[str, Any]:
    value = dataset[()]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    elif isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
        if isinstance(value, bytes):
            value = value.decode("utf-8")
    return json.loads(value)


def _infer_shape_stats(env_meta: Dict[str, Any]) -> Dict[str, int]:
    obs_meta = env_meta.get("obs", {}) or {}
    state_meta = obs_meta.get("state", {}) or {}
    rgb_meta = obs_meta.get("rgb", {}) or {}
    action_meta = env_meta.get("action", {}) or {}
    return {
        "action_dim": int(sum(int(np.prod(shape)) for shape in action_meta.values())),
        "proprio_dim": int(sum(int(np.prod(shape)) for shape in state_meta.values())),
        "num_cameras": int(len(rgb_meta) or 1),
    }


def build_cfg(args) -> Any:
    cfg = get_default_config("Diffusion_CPIQL_DAC")

    with h5py.File(args.demo_path, "r") as f:
        env_meta = _read_json_dataset(f["meta"]["env_meta"])

    stats = _infer_shape_stats(env_meta)
    save_root = os.path.join(args.save_root, args.exp_name)

    cfg.agent_type = "Diffusion_CPIQL_DAC"
    cfg.agent_control_mode = "absolute_joint"
    cfg.device = "cpu"

    cfg.env.library = "piper"
    cfg.env.env_id = "Piper-Dual-Merged"
    cfg.env.env_config_path = args.env_config_path
    cfg.env.env_control_mode = "absolute_joint"
    cfg.env.controller_backend = "joint"
    cfg.env.control_mode = "joint"
    cfg.env.action_dim = stats["action_dim"]
    cfg.env.proprio_dim = stats["proprio_dim"]
    cfg.env.num_cameras = stats["num_cameras"]
    cfg.env.obs_mode = "rgb"
    cfg.env.max_episode_steps = args.max_episode_steps

    cfg.dataset.include_rgb = True
    cfg.dataset.include_depth = False
    cfg.dataset.expert.demo_path = args.demo_path
    cfg.dataset.expert.num_traj = None
    cfg.dataset.expert.format = "auto"

    cfg.train.batch_size = args.batch_size
    cfg.train.num_workers = args.num_workers
    cfg.train.save_interval = args.save_interval

    cfg.actor.action_dim = stats["action_dim"]
    cfg.actor.encoder.proprio_dim = stats["proprio_dim"]
    cfg.actor.encoder.visual.in_channels = 3
    cfg.actor.encoder.visual.backbone_type = "resnet"
    cfg.critic.encoder.proprio_dim = stats["proprio_dim"]
    cfg.critic.encoder.visual.in_channels = 3
    cfg.critic.encoder.visual.backbone_type = "resnet"

    cfg.agent_sp.save_dir = save_root
    cfg.agent_sp.exp_name = args.exp_name
    cfg.agent_sp.actor_dataset_key = "offline"
    cfg.agent_sp.critic_ckpt_path = ""
    cfg.agent_sp.iters = args.actor_iters

    return cfg


def make_dataset(cfg, required_keys):
    return CPIQLExpertDataset(
        cfg=cfg,
        device="cpu",
        required_keys=required_keys,
        h5_path=cfg.dataset.expert.demo_path,
    )


def make_loader(cfg, dataset):
    return DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.train.num_workers,
        pin_memory=(cfg.train.num_workers > 0),
    )


def save_stage_cfg(cfg, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    ConfigManager.save_config(cfg, save_dir=save_dir)


def main():
    parser = argparse.ArgumentParser(description="Train Piper CPIQL-DAC in two stages.")
    parser.add_argument("--demo-path", default="data/merged/dual_merged.h5")
    parser.add_argument("--save-root", default="run_results")
    parser.add_argument("--exp-name", default="piper_dual_merged_cpiql_dac")
    parser.add_argument("--env-config-path", default="agent_infra/Piper_Env/Config/dual_piper_config.yaml")
    parser.add_argument("--critic-iters", type=int, default=200)
    parser.add_argument("--actor-iters", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    args = parser.parse_args()

    cfg = build_cfg(args)
    pipeline_root = os.path.join(args.save_root, args.exp_name)
    critic_dir = os.path.join(pipeline_root, "critic")
    actor_dir = os.path.join(pipeline_root, "actor")
    os.makedirs(critic_dir, exist_ok=True)
    os.makedirs(actor_dir, exist_ok=True)

    print(f"[Config] Saving base config to {pipeline_root}")
    save_stage_cfg(cfg, pipeline_root)

    print("[Stage 1] Build critic agent and dataset")
    critic_agent = make_agent("Diffusion_CPIQL_DAC", cfg)
    critic_dataset = make_dataset(cfg, critic_agent.required_keys)
    critic_loader = make_loader(cfg, critic_dataset)
    save_stage_cfg(cfg, critic_dir)

    print(f"[Stage 1] Train critic for {args.critic_iters} steps")
    critic_agent.train_critic_loop(critic_loader, args.critic_iters, save_dir=critic_dir)
    critic_ckpt = os.path.join(critic_dir, "critic_final.pth")
    critic_agent.save(critic_ckpt, meta={"stage": "critic_final", "demo_path": args.demo_path})

    print("[Stage 2] Build actor agent from critic checkpoint")
    actor_cfg = copy.deepcopy(cfg)
    actor_cfg.agent_sp.critic_ckpt_path = critic_ckpt
    actor_cfg.agent_sp.save_dir = actor_dir
    actor_cfg.agent_sp.exp_name = "actor"
    actor_cfg.agent_sp.iters = args.actor_iters
    save_stage_cfg(actor_cfg, actor_dir)

    actor_agent = make_agent("Diffusion_CPIQL_DAC", actor_cfg)
    actor_dataset = make_dataset(actor_cfg, actor_agent.required_keys)
    actor_payload = {"offline": actor_dataset}

    print(f"[Stage 2] Train actor for {args.actor_iters} steps")
    actor_agent.start_train(actor_payload, additional_args={"phase": "offline"})

    final_actor_ckpt = os.path.join(actor_dir, "actor", "actor_final.pth")
    print(f"[Done] Critic ckpt: {critic_ckpt}")
    print(f"[Done] Actor ckpt: {final_actor_ckpt}")


if __name__ == "__main__":
    main()
