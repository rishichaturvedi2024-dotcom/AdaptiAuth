"""
AdaptiAuth — Adaptive Trust-Scoring Engine Service

Fuses Layer 1 (facial), Layer 2 (liveness), and Layer 3 (behavioral)
scores into a single Trust Score with dynamic re-weighting.
Stub implementation until Phase 4.
"""

from typing import Dict, Optional, Tuple

from app.models.schemas import RiskTier, PolicyAction


class TrustEngine:
    """
    Adaptive Trust-Scoring Engine.

    Phase 4 will implement:
    - Dynamic weighting based on signal confidence/reliability
    - Continuous re-scoring loop (timer + event-driven)
    - Handling of missing/degraded signals
    - Session-level trust history tracking

    NOTE ON IP: The dynamic fusion of continuous multi-modal scores
    (not point-in-time) with confidence-based re-weighting is a key
    novelty claimed in the invention disclosure.
    """

    # Default static weights (Phase 4 will make these dynamic)
    DEFAULT_WEIGHTS = {
        "facial": 0.40,
        "liveness": 0.35,
        "behavioral": 0.25,
    }

    def __init__(self, low_threshold: float = 0.80, medium_threshold: float = 0.50):
        self.low_threshold = low_threshold
        self.medium_threshold = medium_threshold

    def compute_trust_score(
        self,
        facial_score: Optional[float] = None,
        liveness_score: Optional[float] = None,
        behavioral_score: Optional[float] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute fused trust score from layer outputs.
        Returns (trust_score, weights_used).

        TODO (Phase 4): Implement dynamic re-weighting based on
        per-signal confidence and reliability metrics.
        """
        w = weights or self.DEFAULT_WEIGHTS.copy()

        scores = {
            "facial": facial_score,
            "liveness": liveness_score,
            "behavioral": behavioral_score,
        }

        # Handle missing signals: redistribute weight proportionally
        available = {k: v for k, v in scores.items() if v is not None}
        if not available:
            return (0.0, w)

        if len(available) < len(scores):
            total_available_weight = sum(w[k] for k in available)
            if total_available_weight > 0:
                w = {k: w[k] / total_available_weight for k in available}

        trust = sum(scores[k] * w[k] for k in available)
        trust = max(0.0, min(1.0, trust))

        return (round(trust, 4), w)

    def classify_risk(self, trust_score: float) -> Tuple[RiskTier, PolicyAction]:
        """Map a trust score to a risk tier and policy action."""
        if trust_score >= self.low_threshold:
            return (RiskTier.LOW, PolicyAction.ALLOW)
        elif trust_score >= self.medium_threshold:
            return (RiskTier.MEDIUM, PolicyAction.STEP_UP)
        else:
            return (RiskTier.HIGH, PolicyAction.TERMINATE)


# Module-level singleton
trust_engine = TrustEngine()
