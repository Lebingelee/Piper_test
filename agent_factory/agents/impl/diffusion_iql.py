import torch
from agent_factory.agents.base_agent import BaseAgent
from agent_factory.agents.mixins.actor.diffusion import DiffusionActorMixin
from agent_factory.agents.mixins.critic.IQL import IQLCriticMixin
from agent_factory.agents.registry import register_agent

@register_agent("Diffusion_IQL")
class DiffusionIQLAgent(DiffusionActorMixin, IQLCriticMixin, BaseAgent):
    """
    Standard Diffusion + IQL Agent
    """
    def _init_components(self):
        # 1. 初始化 Critic (包含 Critic 自己的 Encoder)
        self._build_critic()
        
        # 2. 初始化 Actor (包含 Actor 自己的 Encoder)
        self._build_actor()
        
        # 注意: 默认两者 Encoder 独立。如果 Config 要求共享，
        # 可以在这里执行 self.actor_encoder = self.critic_encoder 并重建 Actor
    
    def _init_optimizers(self):
        # Mixin 中已经初始化了 self.actor_optimizer, self.q_optimizer, self.v_optimizer
        pass

    def update(self, batch: dict) -> dict: #暂时不考虑使用，建议单独使用self.update_critic(batch)和 self.update_actor(batch)
        self.step += 1
        metrics = {}
        
        # 1. Update Critic
        # IQL Critic 更新 Q 和 V
        critic_metrics = self.update_critic(batch)
        metrics.update(critic_metrics)
        
        # 2. Update Actor
        # 使用当前 batch 数据更新 Policy
        actor_metrics = self.update_actor(batch)
        metrics.update(actor_metrics)
        
        # 3. Target Update (IQL Standard update Q target? In Mixin provided, yes)
        self.soft_update_target()

        return metrics