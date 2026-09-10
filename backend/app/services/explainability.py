"""
AdaptiAuth — Explainability Service

SHAP-based per-feature attributions for the Trust Score.
Stub implementation until Phase 5.
"""

from typing import Dict, List, Optional

from app.models.schemas import SHAPAttribution


class ExplainabilityService:
    """
    SHAP-based explainability for the fused trust score.

    Phase 5 will implement:
    - SHAP KernelExplainer wired to the trust fusion function
    - Per-feature attributions for every trust score computation
    - Consistency validation (attributions sum to score - base_value)
    - Cacheable explanation payloads for the SOC dashboard

    NOTE ON IP: Applying SHAP explainability specifically to the fused,
    continuous, multi-modal trust score is a key novelty.
    """

    def __init__(self):
        self._explainer = None
        # TODO (Phase 5): Initialize SHAP explainer with trust fusion function

    def explain(
        self,
        features: Dict[str, float],
        trust_score: float,
    ) -> List[SHAPAttribution]:
        """
        Generate SHAP attributions for a trust score computation.
        Returns a list of per-feature attributions.

        TODO (Phase 5): Implement real SHAP computation.
        """
        # STUB: Return proportional mock attributions
        base_value = 0.5
        remainder = trust_score - base_value
        n = len(features) if features else 1

        attributions = []
        for feature, value in features.items():
            attr = remainder / n  # evenly distribute for stub
            attributions.append(
                SHAPAttribution(
                    feature=feature,
                    value=value,
                    attribution=round(attr, 4),
                    direction="positive" if attr >= 0 else "negative",
                )
            )

        return attributions

    @property
    def base_value(self) -> float:
        """SHAP base/expected value. TODO (Phase 5): Compute from training data."""
        return 0.5


# Module-level singleton
explainability_service = ExplainabilityService()
