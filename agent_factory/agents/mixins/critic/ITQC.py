import torch
from dataclasses import dataclass
import torch.nn.functional as F
import copy
import numpy as np
from tqdm import tqdm
from typing import Dict, Literal, Optional

# 引用组件
from agent_factory.agents.mixins.critic.base_eval import CriticEvalMixinBase
from agent_factory.agents.mixins.module_builder import ModuleBuilderMixin

from agent_factory.modules.critics.itqc_critic import MultiHeadQuantileNet

from agent_factory.config.structure import BaseCriticConfig


@dataclass
class ITQCCriticConfig(BaseCriticConfig):
    type: str = "itqc"
    lr: float = 2e-4
    num_critics: int = 5
    num_quantiles: int = 5
    mc_loss_weight: float = 0.8
    quantile_huber_kappa: float = 1.0



class ITQCCriticMixin(ModuleBuilderMixin, CriticEvalMixinBase): #Implicit Truncated Q-learning

    CONFIG_CLASS = ITQCCriticConfig
    CONFIG_KEY = "critic"
    REQUIRED_KEYS = {"observations", "action", "next_observations", "reward", "terminated", "discount", "value"}

    def _build_critic(self):
        cfg = self.cfg.critic
        common_args = dict(
            obs_horizon=self.cfg.env.obs_horizon,
            hidden_dims=cfg.hidden_dims,
            num_critics=cfg.num_critics,
            num_quantiles=cfg.num_quantiles
        )

        # === 1. Standard Critic (Normal) ===
        self.critic_encoder = self._build_encoder_from_config(cfg.encoder)
        
        # 假设 action_dim 是单步维度，ITQC MultiHeadQuantileNet 内部会处理 pred_horizon
        self.q_net = MultiHeadQuantileNet(self.critic_encoder, action_dim=self.cfg.env.action_dim, pred_horizon=self.cfg.env.pred_horizon, **common_args)
        self.v_net = MultiHeadQuantileNet(self.critic_encoder, action_dim=0, **common_args)
        self.target_v_net = copy.deepcopy(self.v_net)
        self.target_v_net.requires_grad_(False)

        self.q_optimizer = torch.optim.AdamW(self.q_net.parameters(), lr=cfg.lr)
        self.v_optimizer = torch.optim.AdamW(self.v_net.parameters(), lr=cfg.lr)

        # === 2. Success Critic (Pretty/Suc) ===
        self.suc_critic_encoder = self._build_encoder_from_config(cfg.encoder)
        
        self.suc_q_net = MultiHeadQuantileNet(self.suc_critic_encoder, action_dim=self.cfg.env.action_dim, pred_horizon=self.cfg.env.pred_horizon, **common_args)
        self.suc_v_net = MultiHeadQuantileNet(self.suc_critic_encoder, action_dim=0, **common_args)
        self.suc_target_v_net = copy.deepcopy(self.suc_v_net)
        self.suc_target_v_net.requires_grad_(False)

        self.suc_q_optimizer = torch.optim.AdamW(self.suc_q_net.parameters(), lr=cfg.lr)
        self.suc_v_optimizer = torch.optim.AdamW(self.suc_v_net.parameters(), lr=cfg.lr)

    def soft_update_target(self, critic_type: Literal["Standard", "Pretty"] = "Standard"):
        try:
            tau = self.cfg.agent.soft_update_tau
        except:
            tau = 0.95
        if critic_type == "Standard":
            source, target = self.v_net, self.target_v_net
        else:
            source, target = self.suc_v_net, self.suc_target_v_net
            
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def _tqc_truncation(self, q_atoms, k_drop):
        b, m, n = q_atoms.shape
        atoms_flat = q_atoms.reshape(b, -1)
        sorted_atoms, _ = torch.sort(atoms_flat, dim=1)
        n_keep = (m * n) - k_drop
        return sorted_atoms[:, :n_keep]

    # --- Loss Functions (Optimized) ---
    @staticmethod
    def quantile_squared_loss(pred, target, tau):
        """
        [Optimized] 计算分位数平方损失。
        pred: [B, num_critics, num_quantiles]
        target: [B, 1] (MC) 或 [B, N_keep] (Distill)
        tau: [num_quantiles]
        """
        b, m, n = pred.shape
        tau_view = tau.view(1, 1, n).to(pred.device) # [1, 1, n]

        if target.shape[-1] == 1:
            # Point 2: MC Loss Fast Path (针对标量目标，避免 4D 广播)
            # error shape: [B, M, N]
            error = target.view(b, 1, 1) - pred
            # Point 4: torch.where 直接计算权重，减少中间张量复制
            weight = torch.where(error < 0, 1 - tau_view, tau_view)
            return weight * (error ** 2)
        else:
            # Point 3: Pairwise Flattening (展平分位数维度进行成对计算)
            pred_flat = pred.reshape(b, -1, 1) # [B, M*N, 1]
            target_flat = target.view(b, 1, -1) # [B, 1, N_keep]
            error = target_flat - pred_flat    # [B, M*N, N_keep]
            
            # 对齐 tau 权重 (每个 N 个原子对应同一个分位点)
            tau_repeat = tau.view(1, 1, n, 1).expand(b, m, n, 1).reshape(b, -1, 1).to(pred.device)
            weight = torch.where(error < 0, 1 - tau_repeat, tau_repeat)
            
            # 还原为原有的 [B, M, N] 形状
            return (weight * (error ** 2)).mean(dim=-1).view(b, m, n)

    @staticmethod
    def quantile_huber_loss(pred, target, tau, kappa=1.0):
        """
        [Optimized] 计算分位数 Huber 损失。
        pred: [B, num_critics, num_quantiles]
        target: [B, M*N] (Bellman Target)
        """
        b, m, n = pred.shape
        pred_flat = pred.reshape(b, -1, 1)
        target_flat = target.view(b, 1, -1)
        error = target_flat - pred_flat # [B, M*N, Atoms]

        # Huber Kernel
        abs_err = error.abs()
        huber = torch.where(abs_err < kappa, 0.5 * error.pow(2), kappa * (abs_err - 0.5 * kappa))

        # Weight (Point 4)
        tau_repeat = tau.view(1, 1, n, 1).expand(b, m, n, 1).reshape(b, -1, 1).to(pred.device)
        weight = torch.where(error < 0, 1 - tau_repeat, tau_repeat)
        
        # 还原为 [B, M, N], 对 target atoms 求和
        return (weight * huber).sum(dim=-1).view(b, m, n)

    """
    # --- Original Implementation (For Observation) ---
    @staticmethod
    def quantile_squared_loss_ORIGINAL(pred, target, tau):
        pred_expanded = pred.unsqueeze(-1)
        if target.ndim == 1: target = target.unsqueeze(-1)
        if target.ndim == 2: target_expanded = target.unsqueeze(1).unsqueeze(1)
        else: raise ValueError(f"Target dim {target.ndim} invalid")
        
        tau_expanded = tau.view(1, 1, -1, 1).to(pred.device)
        error = target_expanded - pred_expanded
        weight = torch.abs(tau_expanded - (error < 0).float())
        return (weight * (error ** 2)).mean(dim=-1)

    @staticmethod
    def quantile_huber_loss_ORIGINAL(pred, target, tau, kappa=1.0):
        pred_expanded = pred.unsqueeze(-1)
        target_expanded = target.unsqueeze(1).unsqueeze(1)
        tau_expanded = tau.view(1, 1, -1, 1).to(pred.device)
        error = target_expanded - pred_expanded
        
        abs_err = error.abs()
        huber = torch.where(abs_err < kappa, 0.5 * error.pow(2), kappa * (abs_err - 0.5 * kappa))
        weight = torch.abs(tau_expanded - (error < 0).float())
        return (weight * huber).sum(dim=-1)
    """

    def _prepare_batch_for_itqc(self, batch):
        # 复用 IQL 里的 _preprocess_obs (假设 Agent 提供)
        obs = self._preprocess_obs(batch["observations"])
        for k in obs: obs[k] = obs[k].to(self.device)
        
        next_obs = self._preprocess_obs(batch["next_observations"])
        for k in next_obs: next_obs[k] = next_obs[k].to(self.device)
        
        actions = batch["action"].to(self.device)
        
        rewards = batch["reward"].to(self.device)
        if rewards.ndim == 1: rewards = rewards.unsqueeze(-1)
        # 归一化 (根据配置)
        scale = getattr(self.cfg.env, 'reward_scale', 100.0) 
        rewards = rewards / scale

        dones = batch["terminated"].to(self.device).float()
        if dones.ndim == 1:
            dones = dones.unsqueeze(-1)

        discounts = batch.get("discount", None)
        if discounts is not None:
            discounts = discounts.to(self.device).float()
            if discounts.ndim == 1:
                discounts = discounts.unsqueeze(-1)
        else:
            discounts = (1.0 - dones) * float(self.cfg.env.gamma)

        # MC Values
        mc_values = batch["value"].to(self.device) # GT from dataset
        if mc_values.ndim == 1:
            mc_values = mc_values.unsqueeze(-1)
        mc_values = mc_values / scale

        return obs, actions, rewards, next_obs, dones, discounts, mc_values

    def _distribution_mean(self, values: torch.Tensor) -> torch.Tensor:
        return values.mean(dim=(1, 2), keepdim=False).unsqueeze(-1)

    def compute_loss_critic(
        self, 
        obs: torch.Tensor, 
        actions: torch.Tensor, 
        rewards: torch.Tensor, 
        next_obs: torch.Tensor, 
        dones: torch.Tensor,
        discounts: torch.Tensor,
        mc_returns: torch.Tensor,
        critic_type: Literal["Standard", "Pretty"] = "Standard"
            ) :
        """
        计算 Critic (V-Net 和 Q-Net) 的 Loss。
        参数对应 RL 中的 (s, a, r, s', d) 以及 MC 回报。
        """
        # 1. 切换网络上下文 (根据 critic_type 选择对应的网络实例)
        if critic_type == "Standard":
            q_net, v_net = self.q_net, self.v_net
            target_v_net = self.target_v_net
        else:
            q_net, v_net = self.suc_q_net, self.suc_v_net
            target_v_net = self.suc_target_v_net
        
        tau = v_net.tau

        # --- Part 1: Value Loss Calculation ---
        v_pred = v_net(obs)
        
        with torch.no_grad():
            # Q-Net 在 V Loss 中仅作为 Target，不需要梯度
            q_full = q_net(obs, actions)
            # Drop Top-K atoms (TQC truncate)
            q_truncated = self._tqc_truncation(q_full, k_drop=v_net.num_critics) 
        
        # ITQC 混合 Loss: MC Regression + Distributional Distillation
        loss_mc = self.quantile_squared_loss(v_pred, mc_returns, tau)
        loss_distill = self.quantile_squared_loss(v_pred, q_truncated, tau)
        
        w = self.cfg.critic.mc_loss_weight
        # Sum over quantiles/atoms, mean over batch
        loss_v = (w * loss_mc + (1.0 - w) * loss_distill).sum(dim=-1).mean()

        # --- Part 2: Q Loss Calculation ---
        # Q-Net 需要梯度
        q_pred = q_net(obs, actions)
        
        with torch.no_grad():
            next_v_pred = target_v_net(next_obs)
            # Flatten atoms: [B, M, N] -> [B, M*N]
            next_v_flat = next_v_pred.reshape(rewards.shape[0], -1)
            # Distributional Bellman Update
            del dones
            target_q_atoms = rewards + discounts * next_v_flat
        
        loss_q_map = self.quantile_huber_loss(q_pred, target_q_atoms, tau, kappa=self.cfg.critic.quantile_huber_kappa)
        loss_q = loss_q_map.mean()

        # --- Metrics ---
        with torch.no_grad():
            # 计算 Advantage 用于日志监控: Q_mean - V_mean
            # v_pred[..., 2:] 是为了避开 quantile 边缘的不稳定值 (经验性操作，可视情况调整)
            adv = q_pred.mean() - v_pred.mean()

        return loss_v, loss_q, adv.item()

    def update_critic(self, batch: Dict, critic_type: Literal["Standard", "Pretty"] = "Standard") -> Dict:
        """
        从 Batch 中提取数据，调用计算逻辑，并执行梯度更新。
        """
        # 1. 准备数据 (解包 Batch -> Tensors)
        # 确保 _prepare_batch_for_itqc 返回的顺序与 compute_loss_critic 参数一致
        obs, actions, rewards, next_obs, dones, discounts, mc_returns = self._prepare_batch_for_itqc(batch)
        
        # 2. 计算 Loss
        loss_v, loss_q, adv = self.compute_loss_critic(
            obs, actions, rewards, next_obs, dones, discounts, mc_returns, critic_type
        )

        # 3. 获取优化器
        if critic_type == "Standard":
            optimizer_q, optimizer_v = self.q_optimizer, self.v_optimizer
        else:
            optimizer_q, optimizer_v = self.suc_q_optimizer, self.suc_v_optimizer

        # 4. 反向传播与更新 (V Net)
        optimizer_v.zero_grad()
        optimizer_q.zero_grad()

        loss_v.backward()
        loss_q.backward()
        
        optimizer_v.step()
        optimizer_q.step()

        return {
            f"loss_v": loss_v.item(), 
            f"loss_q": loss_q.item(), 
            "adv": adv
        }

    @torch.no_grad()
    def eval_batch(self, batch: Dict, only_obs: bool = True) -> Dict[str, torch.Tensor]:
        obs = self._preprocess_obs(batch["observations"])

        v_std = self._distribution_mean(self.v_net(obs))
        v_suc = self._distribution_mean(self.suc_v_net(obs))
        gap = v_suc - v_std

        results = {
            "figure:value/V(k=0)": v_std.squeeze(-1),
            "figure:value/V(k=1)": v_suc.squeeze(-1),
            "figure:value/critic_gap": gap.squeeze(-1),
        }

        if not only_obs:
            actions = batch["action"].to(self.device)
            q_std = self._distribution_mean(self.q_net(obs, actions))
            q_suc = self._distribution_mean(self.suc_q_net(obs, actions))
            adv_std = q_std - v_std

            results["figure:action_value/Q(k=0)"] = q_std.squeeze(-1)
            results["figure:action_value/Q(k=1)"] = q_suc.squeeze(-1)
            results["figure:action_value/adv(k=0)"] = adv_std.squeeze(-1)

        if "frame" in batch:
            results["frame"] = batch["frame"]
        return results

    def relabel_data(self, dataset, phase="pretrain", critic_type="normal"):
        """ITQC Relabel: Adv = Mean(Q) - Mean(V_filtered)"""
        print(f"[ITQC Relabel] Phase: {phase}, Type: {critic_type}")
        self.eval()
        
        # 选择网络
        net_source = self.q_net if critic_type == "normal" else self.suc_q_net
        v_source = self.v_net if critic_type == "normal" else self.suc_v_net

        loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
        all_advs = []
        
        with torch.no_grad():
            for batch in loader:
                obs = self._preprocess_obs(batch["observations"])
                for k in obs: obs[k] = obs[k].to(self.device)
                actions = batch["action"].to(self.device)
                
                # Q Mean
                q_out = net_source(obs, actions)
                q_mean = q_out.mean(dim=(1, 2)).unsqueeze(-1)
                
                # V Filtered Mean (Top quantiles usually, indices 2,3,4)
                v_out = v_source(obs)
                v_filtered = v_out[..., 2:].mean(dim=(1, 2)).unsqueeze(-1)
                
                adv = q_mean - v_filtered
                all_advs.append(adv.cpu().numpy())
                
        self.train()
        all_advs = np.concatenate(all_advs)

        if phase == "pretrain":
            cfg_sp = self.cfg.agent_sp
            quantile = cfg_sp.get("threshold_quantile_offline", 0.8)
        else:
            cfg_sp = self.cfg.agent_sp
            quantile = cfg_sp.get("threshold_quantile_online", 0.3)
        
        threshold = np.percentile(all_advs, (1.0 - quantile) * 100)
        
        is_positive = (all_advs.squeeze() > threshold)
        new_conds = torch.where(
            torch.from_numpy(is_positive).to(self.device),
            torch.tensor(1.0, device=self.device),
            torch.tensor(-1.0, device=self.device)
        )
        dataset.conds = new_conds.float().cpu() # Ensure it's on CPU for multiprocessing
        print(f"[{critic_type}] Threshold: {threshold:.4f}, Pos Ratio: {is_positive.mean():.2%}")
