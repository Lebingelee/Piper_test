from typing import Any, Dict

import torch

from agent_factory.agents.impl.cpiql_only import CPIQLOnlyAgent
from agent_factory.agents.registry import register_agent


@register_agent("CPIQL_Ratio_Only")
class CPIQLRatioOnlyAgent(CPIQLOnlyAgent):
    """
    Critic-only CPIQL agent with a ratio risk score.

    Training/loading behavior is inherited from CPIQL_Only. Only the exported
    phase-1 risk signal changes from DeltaV to (V(k=1) - V(k=0)) / V(k=1).
    """

    @torch.no_grad()
    def eval_risk_batch(self, batch: Dict[str, Any], only_obs: bool = True) -> Dict[str, Any]:
        outputs = self.eval_batch(batch, only_obs=only_obs)
        outputs = dict(outputs)
        v_k0 = outputs["figure:value/V(k=0)"]
        v_k1 = outputs["figure:value/V(k=1)"]
        delta_v = v_k1 - v_k0
        eps = 1e-6
        agent_sp = getattr(self.cfg, "agent_sp", None)
        if agent_sp is not None:
            try:
                if "ratio_epsilon" in agent_sp:
                    eps = float(agent_sp.ratio_epsilon)
            except (AttributeError, TypeError):
                pass
        denom = torch.clamp(v_k1, min=eps)
        raw_ratio = delta_v / denom
        ratio = torch.clamp(raw_ratio, min=0.0)
        outputs.update(
            {
                "risk_score": ratio,
                "risk_signal/raw_v_k0": v_k0,
                "risk_signal/raw_v_k1": v_k1,
                "risk_signal/raw_delta_v": delta_v,
                "risk_signal/raw_cpiql_ratio": raw_ratio,
                "risk_signal/score_formula": "max_v_k1_minus_v_k0_over_v_k1_zero",
                "risk_signal/method_id": "cpiql_ratio",
                "risk_signal/display_name": "CPIQL Ratio",
            }
        )
        if "figure:action_value/Q(k=0)" in outputs:
            outputs["risk_signal/raw_q_k0"] = outputs["figure:action_value/Q(k=0)"]
        if "figure:action_value/Q(k=1)" in outputs:
            outputs["risk_signal/raw_q_k1"] = outputs["figure:action_value/Q(k=1)"]
        if "figure:action_value/adv(k=0)" in outputs:
            outputs["risk_signal/raw_adv_k0"] = outputs["figure:action_value/adv(k=0)"]
        return outputs
    
    """
    @torch.no_grad()
    def eval_risk_batch(self, batch: Dict[str, Any], only_obs: bool = True) -> Dict[str, Any]:
        outputs = self.eval_batch(batch, only_obs=only_obs)
        outputs = dict(outputs)
        v_k0 = outputs["figure:value/V(k=0)"]
        v_k1 = outputs["figure:value/V(k=1)"]
        delta_v = v_k1 - v_k0
        eps = 1e-6
        agent_sp = getattr(self.cfg, "agent_sp", None)
        if agent_sp is not None:
            try:
                if "ratio_epsilon" in agent_sp:
                    eps = float(agent_sp.ratio_epsilon)
            except (AttributeError, TypeError):
                pass
        denom = torch.clamp(v_k1, min=eps)
        ratio = delta_v / denom
        outputs.update(
            {
                "risk_score": ratio,
                "risk_signal/raw_v_k0": v_k0,
                "risk_signal/raw_v_k1": v_k1,
                "risk_signal/raw_delta_v": delta_v,
                "risk_signal/raw_cpiql_ratio": ratio,
                "risk_signal/score_formula": "v_k1_minus_v_k0_over_v_k1",
                "risk_signal/method_id": "cpiql_ratio",
                "risk_signal/display_name": "CPIQL Ratio",
            }
        )
        if "figure:action_value/Q(k=0)" in outputs:
            outputs["risk_signal/raw_q_k0"] = outputs["figure:action_value/Q(k=0)"]
        if "figure:action_value/Q(k=1)" in outputs:
            outputs["risk_signal/raw_q_k1"] = outputs["figure:action_value/Q(k=1)"]
        if "figure:action_value/adv(k=0)" in outputs:
            outputs["risk_signal/raw_adv_k0"] = outputs["figure:action_value/adv(k=0)"]
        return outputs

        """
