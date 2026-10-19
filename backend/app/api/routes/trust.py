"""
AdaptiAuth — Trust Score Routes

Endpoints for querying trust scores, triggering re-scoring, and
retrieving SHAP explanations. Placeholder stubs until the fusion
engine and explainability layer are implemented in Phases 4–5.
"""

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, status

from app.models.schemas import (
    TrustScoreResult,
    ExplainedTrustScore,
    RiskTier,
    PolicyAction,
    SHAPAttribution,
)

from ml.fusion.trust_engine import TrustEngine
from ml.explainability.shap_explainer import TrustSHAPExplainer
from app.services.policy import PolicyEngine
from app.services.live_state import live_signal_store

router = APIRouter(prefix="/trust", tags=["Trust Scoring"])
trust_engine = TrustEngine()
shap_explainer = TrustSHAPExplainer(trust_engine)
policy_engine = PolicyEngine()


@router.get("/{session_id}/score", response_model=TrustScoreResult)
async def get_trust_score(session_id: str):
    """
    Get the current trust score for a session.
    """
    # For prototype, we mock fetching the latest score.
    # It shares logic with rescore for simplicity in this stub.
    return await trigger_rescore(session_id)


@router.post("/{session_id}/rescore", response_model=TrustScoreResult)
async def trigger_rescore(session_id: str):
    """
    Trigger an immediate re-scoring of the session.
    """
    signals = live_signal_store.mark_stale_faces_absent(session_id)
    facial_s, facial_c = signals.facial_score, signals.facial_confidence
    live_s, live_c = signals.liveness_score, signals.liveness_confidence
    beh_s, beh_c = signals.behavioral_score, signals.behavioral_confidence
    
    result = trust_engine.compute_trust_score(
        facial_score=facial_s,
        facial_conf=facial_c,
        liveness_score=live_s,
        liveness_conf=live_c,
        behavioral_score=beh_s,
        behavioral_conf=beh_c,
    )
    
    t_score = result["trust_score"]
    
    # Policy evaluation
    policy_res = policy_engine.evaluate(t_score)
    tier = policy_res["risk_tier"]
    action = policy_res["policy_action"]
    
    # Execute policy side effects (stubbed)
    policy_engine.execute_action(session_id, action)
        
    return TrustScoreResult(
        trust_score=t_score,
        facial_score=facial_s,
        liveness_score=live_s,
        behavioral_score=beh_s,
        weights=result["weights"],
        risk_tier=tier,
        policy_action=action,
        timestamp=datetime.now(timezone.utc),
    )


@router.get("/{session_id}/explain", response_model=ExplainedTrustScore)
async def get_trust_explanation(session_id: str):
    """
    Get the SHAP-explained trust score for a session.
    """
    signals = live_signal_store.mark_stale_faces_absent(session_id)
    facial_s, facial_c = signals.facial_score, signals.facial_confidence
    live_s, live_c = signals.liveness_score, signals.liveness_confidence
    beh_s, beh_c = signals.behavioral_score, signals.behavioral_confidence
    
    # Recompute to get the score and policy
    result = trust_engine.compute_trust_score(
        facial_score=facial_s,
        facial_conf=facial_c,
        liveness_score=live_s,
        liveness_conf=live_c,
        behavioral_score=beh_s,
        behavioral_conf=beh_c,
    )
    t_score = result["trust_score"]
    
    # Policy evaluation
    policy_res = policy_engine.evaluate(t_score)
    tier = policy_res["risk_tier"]
    action = policy_res["policy_action"]
        
    # Get SHAP explanations
    raw_attributions = shap_explainer.explain(
        facial_s, facial_c, live_s, live_c, beh_s, beh_c
    )
    
    attributions = [
        SHAPAttribution(
            feature=attr["feature"],
            value=attr["value"],
            attribution=attr["attribution"],
            direction=attr["direction"]
        ) for attr in raw_attributions
    ]

    return ExplainedTrustScore(
        trust_score=t_score,
        risk_tier=tier,
        policy_action=action,
        base_value=shap_explainer.base_value,
        attributions=attributions,
        signals={
            "facial_score": facial_s,
            "facial_confidence": facial_c,
            "liveness_score": live_s,
            "liveness_confidence": live_c,
            "behavioral_score": beh_s,
            "behavioral_confidence": beh_c,
            "keystroke_score": signals.keystroke_score,
            "mouse_score": signals.mouse_score,
            "context_score": signals.context_score,
        },
        weights=result["weights"],
        updated_at=signals.updated_at,
        timestamp=datetime.now(timezone.utc),
    )
