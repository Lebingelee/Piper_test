"""π₀.₅ agent registration for deterministic VLM-prefix feature export."""

from dataclasses import dataclass

from agent_factory.agents.base_agent import BaseAgent
from agent_factory.agents.mixins.actor.pi05 import Pi05ActorMixin
from agent_factory.agents.registry import register_agent


@dataclass
class Pi05AgentSpecialConfig:
    """Experiment metadata only; policy schema belongs to ``actor``."""

    save_dir: str = "run_results"
    exp_name: str = "pi05"


class MainMixin:
    CONFIG_CLASS = Pi05AgentSpecialConfig
    CONFIG_KEY = "agent_sp"


@register_agent("pi05")
class Pi05VanillaAgent(MainMixin, Pi05ActorMixin, BaseAgent):
    """Minimal agent_factory host for LeRobot π₀.₅ feature extraction."""

    def _init_components(self):
        self._build_actor()

    def _init_optimizers(self):
        pass

    def start_train(self, dataset, additional_args=None):
        del dataset, additional_args
        raise NotImplementedError(
            "π₀.₅ training is performed by LeRobot after H5 conversion; "
            "agent_factory does not implement it in this migration."
        )
