from typing import Any, Dict


class CriticEvalMixinBase:
    """
    Lightweight critic-side evaluation interface.

    Concrete critic mixins may implement `eval_batch()` to expose algorithm-
    specific metrics for offline visualization without coupling the script to a
    particular critic head layout.
    """

    def eval_batch(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement critic eval_batch()."
        )

    def eval_risk_batch(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Batch-level continuous risk signal interface for offline export.

        `eval_batch()` remains the critic diagnostics/visualization interface.
        Concrete agent impls should override this method when they want to
        declare a canonical risk score. Deployment-time boolean decisions still
        belong to BaseAgent.get_risk().
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement critic eval_risk_batch()."
        )
