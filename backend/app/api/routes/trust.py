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
    # For prototype without continuous signal DB, we'll mock the latest signals
    # In reality, this would fetch the last 1-3 seconds of signals from DB
    facial_s, facial_c = 0.90, 0.95
    live_s, live_c = 0.88, 0.90
    beh_s, beh_c = 0.78, 0.85
    
    result = trust_engine.compute_trust_score(
        facial_s, facial_c,
        live_s, live_c,
        beh_s, beh_c
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
    # For prototype, we mock fetching the latest signals from DB
    facial_s, facial_c = 0.90, 0.95
    live_s, live_c = 0.88, 0.90
    beh_s, beh_c = 0.78, 0.85
    
    # Recompute to get the score and policy
    result = trust_engine.compute_trust_score(
        facial_s, facial_c, live_s, live_c, beh_s, beh_c
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
        timestamp=datetime.now(timezone.utc),
    )
