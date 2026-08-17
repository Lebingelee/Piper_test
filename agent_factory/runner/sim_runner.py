from typing import Any, Dict

from omegaconf import DictConfig

from agent_factory.runner.base_runner import BaseRunner
from agent_factory.runner.config_utils import runner_config_get


class SimRunner(BaseRunner):
    """
    Simulation rollout runner.

    The base runner owns async policy inference, action dispatch, H5 writing, and
    FIFO cleanup. SimRunner only adds simulation-specific episode semantics:
    robosuite writes task success into ``info["success"]`` while
    ``terminated`` depends on the environment's terminate_on_success flag.
    """

    def __init__(self, cfg: DictConfig, agent: Any, env: Any):
        super().__init__(cfg=cfg, agent=agent, env=env)
        self.stop_on_success = bool(runner_config_get(cfg, "stop_on_success", True))
        if runner_config_get(cfg, "min_save_steps", None) is None:
            try:
                self.cfg.runner.config.min_save_steps = 1
            except Exception:
                pass

    def _extract_success(self, info: Dict[str, Any], terminated: bool) -> bool:
        for key in ("success", "is_success"):
            if key in info:
                return self._any_true(info[key])
        return False

    def _normalize_step_flags(
        self,
        info: Dict[str, Any],
        terminated: bool,
        truncated: bool,
        success: bool,
    ):
        success = bool(success)
        if success and self.stop_on_success:
            return True, False, True
        return bool(terminated), bool(truncated), success
