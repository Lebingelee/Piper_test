import torch
from ..normalization_mixins import ActionNormMixin
from agent_factory.modules.actors.diffusion import VanillaDiffusionPolicy, ConditionalDiffusionPolicy
from agent_factory.modules.encoders.state_encoder import BaseStateEncoder
from agent_factory.modules.encoders.visual_encoder import VisualEncoder
from agent_factory.config.structure import DiffusionActorConfig


class DiffusionActorMixin(ActionNormMixin):
    """
    Mixin: 为 Agent 提供 Diffusion Policy 的构建、训练和推理能力。
    """

    CONFIG_CLASS = DiffusionActorConfig
    CONFIG_KEY = "actor"
    # 声明：我训练 Diffusion 需要这些数据
    REQUIRED_KEYS = {"observations", "action"}


    def _build_actor(self):
        """
        根据 cfg.actor.type ('vanilla' or 'conditional') 初始化 Actor
        """
        cfg: DiffusionActorConfig = self.cfg.actor
        
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
            proprio_dim=self.cfg.env.proprio_dim,
            out_dim=cfg.encoder.out_dim,
            num_cameras=self.cfg.env.num_cameras
        )
        
        common_args = dict(
            state_encoder=self.actor_encoder,
            action_dim=self.cfg.env.action_dim, #注意这里是从环境中获取
            pred_horizon=self.cfg.env.pred_horizon,
            obs_horizon=self.cfg.env.obs_horizon,
            unet_config=cfg.unet,
            scheduler_config=cfg.scheduler
        )

        # 2. 实例化 Policy
        if cfg.use_extra_cond:
            self.actor = ConditionalDiffusionPolicy(
                cond_dim=cfg.cond_dim,
                cond_embed_dim=cfg.cond_embed_dim,
                **common_args
            )
        else:
            self.actor = VanillaDiffusionPolicy(**common_args)
        

        # 3. 优化器
        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(), 
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
        )

    def update_actor(self, batch: dict) -> dict:
        """
        从 Batch 中提取数据并更新 Actor
        """
        # CRITICAL: 必须调用预处理，将 uint8 图像转换为 float32 并归一化
        obs = self._preprocess_obs(batch['observations']) 
        
        # --- 归一化动作 ---
        actions = self.normalize_action(batch['action'])
        
        # 根据类型决定是否传 cond
        if isinstance(self.actor, ConditionalDiffusionPolicy):
            cond = batch.get('cond', None)
            if cond is None:
                # 如果是 Conditional Policy 但 Batch 没 cond，报错或给默认
                raise ValueError("ConditionalDiffusionPolicy requires 'cond' in batch.")
            
            # 这里调用的是 actor.forward
            loss = self.actor(
                obs, actions, cond, 
                use_cfg_loss=getattr(self.cfg.actor, "use_cfg_loss", False),
                cfg_drop_rate=getattr(self.cfg.actor, "cfg_drop_rate", 0.1)
            )
        else:
            # VanillaDiffusionPolicy.forward(obs, actions)
            loss = self.actor(obs, actions)

        # Backprop
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return {"loss_actor": loss.item()}

    def sample_action(self, obs, initial_noise=None):
        """
        推理接口
        """
        was_training = self.actor.training
        self.actor.eval()
        # 🟢 加固逻辑：自动处理归一化和设备搬运，兼容 uint8 和 float32
        obs = self._preprocess_obs(obs)
        try:
            with torch.no_grad():
                if initial_noise is not None:
                    initial_noise = torch.as_tensor(initial_noise, dtype=torch.float32, device=self.device)
                norm_action = self.actor.sample_action(obs, initial_noise=initial_noise)
                # --- 反归一化动作 ---
                return self.denormalize_action(norm_action)
        finally:
            if was_training:
                self.actor.train()
