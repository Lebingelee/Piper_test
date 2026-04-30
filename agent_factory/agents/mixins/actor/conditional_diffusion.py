import torch
from ..normalization_mixins import ActionNormMixin
from agent_factory.modules.actors.diffusion import ConditionalDiffusionPolicy
from agent_factory.modules.encoders.state_encoder import BaseStateEncoder
from agent_factory.modules.encoders.visual_encoder import VisualEncoder
from agent_factory.config.structure import Conditional_DiffusionActorConfig


class ConditionalDiffusionActorMixin(ActionNormMixin):
    """
    Mixin: 为 Agent 提供 Diffusion Policy 的构建、训练和推理能力。
    """

    CONFIG_CLASS = Conditional_DiffusionActorConfig
    CONFIG_KEY = "actor"
    # 声明：我训练 Diffusion 需要这些数据
    REQUIRED_KEYS = {"observations", "action", "cond"}


    def _build_actor(self):
        """
        根据 cfg.actor.type ('vanilla' or 'conditional') 初始化 Actor
        """
        cfg: Conditional_DiffusionActorConfig = self.cfg.actor
        
        # --- 初始化归一化器 ---
        self._init_action_normalizer()

        # 1. 准备 Encoder (Actor 独享)
        # 注意：这里假设 Config 结构已经对齐
        visual_encoder = VisualEncoder(
            in_channels=cfg.encoder.visual.in_channels,
            out_dim=cfg.encoder.visual.out_dim,
            backbone_type=cfg.encoder.visual.backbone_type,
            pool_feature_map=cfg.encoder.visual.pool_feature_map,
            use_group_norm=cfg.encoder.visual.use_group_norm
        )
        
        self.actor_encoder = BaseStateEncoder(
            visual_encoder=visual_encoder,
            proprio_dim=cfg.encoder.proprio_dim,
            out_dim=cfg.encoder.out_dim,
            num_cameras=self.cfg.env.num_cameras,
            view_fusion=cfg.encoder.view_fusion
        )
        
        common_args = dict(
            state_encoder=self.actor_encoder,
            action_dim=self.cfg.env.action_dim, #注意这里是从环境中获取
            pred_horizon=self.cfg.env.pred_horizon,
            obs_horizon=self.cfg.env.obs_horizon,
            unet_config=cfg.unet,
            scheduler_config=cfg.scheduler
        )

        self.actor = ConditionalDiffusionPolicy(
            cond_dim=cfg.cond_dim,
            cond_embed_dim=cfg.cond_embed_dim,
            
            **common_args
        )
       
        # 3. 优化器
        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(), 
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
        )
    def compute_loss_actor(self, obs_seq, action_seq) -> torch.Tensor:
        """
        计算 Actor 的 Diffusion Loss，支持 CFG Loss
        """
        import torch.nn.functional as F
        B = obs_seq["state"].shape[0]
        
        # 1. 编码
        obs_seq = self._preprocess_obs(obs_seq)

        obs_feat = self.actor.state_encoder(obs_seq)
        obs_feat = obs_feat.flatten(start_dim=1)

        # 2. CFG Condition
        cond = obs_seq["cond"].float().to(self.device)
        # CFG Dropout
        drop_mask = (torch.rand(B, device=self.device) < self.cfg.actor.cfg_drop_rate).unsqueeze(1)
        train_cond = cond.clone()
        train_cond[drop_mask] = 0.0 
        
        cond_feat = self.actor.cond_embed_net(train_cond)
        global_cond = torch.cat([obs_feat, cond_feat], dim=1)
        
        # Uncond 特征 (用于 CFG Loss 计算)
        uncond = torch.zeros_like(cond)
        uncond_feat = self.actor.cond_embed_net(uncond)
        global_uncond = torch.cat([obs_feat, uncond_feat], dim=1)
        
        # 3. 加噪
        # --- 使用 Mixin 提供的 normalize_action ---
        norm_action_seq = self.normalize_action(action_seq)
        
        noise = torch.randn((B, self.actor.pred_horizon, self.cfg.env.action_dim), device=self.device)
        timesteps = torch.randint(0, self.actor.noise_scheduler.config.num_train_timesteps, (B,), device=self.device).long()
        noisy_actions = self.actor.noise_scheduler.add_noise(norm_action_seq, noise, timesteps)

        # 4. 预测与 Loss
        noise_pred = self.actor.noise_pred_net(noisy_actions, timesteps, global_cond=global_cond)

        if self.cfg.actor.CFG_alpha == 0.0:
            return F.mse_loss(noise_pred, noise)

        noise_pred_uncond = self.actor.noise_pred_net(noisy_actions, timesteps, global_cond=global_uncond)
        loss_uncond = F.mse_loss(noise_pred_uncond, noise)
        loss_cond = F.mse_loss(noise_pred, noise)
        
        return loss_uncond + self.cfg.actor.CFG_alpha * loss_cond
    def update_actor(self, batch: dict) -> dict:
        """
        从 Batch 中提取数据并更新 Actor
        """
        obs_seq = batch["observations"]
        for k in obs_seq: obs_seq[k] = obs_seq[k].to(self.device)
        if "cond" in batch: 
            obs_seq["cond"] = batch["cond"].to(self.device)
        action_seq = batch["action"].to(self.device)
        
        loss_actor = self.compute_loss_actor(obs_seq, action_seq)
        
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()
        

        return {"loss_actor": loss_actor.item()}

    def sample_action(self, obs, cond=None, initial_noise=None):#一般只用来推理
        """
        推理接口
        """
        self.actor.eval()
        # 🟢 加固逻辑：调用 Agent 基类的统一预处理
        obs = self._preprocess_obs(obs)
        
        with torch.no_grad():
            if isinstance(self.actor, ConditionalDiffusionPolicy):
                # 默认使用配置中的 inference guidance scale
                scale = getattr(self.cfg.actor, "guidance_scale", 1.0)
                if cond is None:
                    # 如果未提供外部 cond，尝试从 obs 字典中获取
                    cond = obs.get("cond", torch.ones((obs['state'].shape[0], 1), device=self.device))
                else:
                    # 如果提供了外部 cond，确保搬运到设备
                    if not isinstance(cond, torch.Tensor):
                        cond = torch.as_tensor(cond, dtype=torch.float32)
                    cond = cond.to(self.device)
                if cond.dim() == 1:
                    cond = cond.unsqueeze(-1)
                cond = cond.float()

                if initial_noise is not None:
                    initial_noise = torch.as_tensor(initial_noise, dtype=torch.float32, device=self.device)
                norm_action = self.actor.sample_action(
                    obs,
                    cond,
                    guidance_scale=scale,
                    initial_noise=initial_noise,
                )
            else:
                if initial_noise is not None:
                    initial_noise = torch.as_tensor(initial_noise, dtype=torch.float32, device=self.device)
                norm_action = self.actor.sample_action(obs, initial_noise=initial_noise)
            
            # --- 反归一化动作 ---
            return self.denormalize_action(norm_action)
        self.actor.train()
