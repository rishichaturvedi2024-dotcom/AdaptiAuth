"""
AdaptiAuth — SOC Dashboard Routes

Endpoints for the Security Operations Center (SOC) dashboard:
session list, trust event timeline, alerts, and summary stats.
"""

from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, HTTPException, status

from app.core.database import get_db
from app.models.schemas import (
    AlertListResponse,
    AlertResponse,
    AlertType,
    DashboardSummary,
    TrustEventResponse,
    RiskTier,
    PolicyAction,
)

router = APIRouter(prefix="/dashboard", tags=["SOC Dashboard"])


@router.get("/summary", response_model=DashboardSummary)
async def get_dashboard_summary():
    """Get summary statistics for the SOC dashboard."""
    with get_db() as conn:
        active = conn.execute(
            "SELECT COUNT(*) as cnt FROM sessions WHERE status = 'active'"
        ).fetchone()["cnt"]

        total_alerts = conn.execute(
            "SELECT COUNT(*) as cnt FROM alerts"
        ).fetchone()["cnt"]

        unack = conn.execute(
            "SELECT COUNT(*) as cnt FROM alerts WHERE acknowledged = 0"
        ).fetchone()["cnt"]

        avg_row = conn.execute(
            "SELECT AVG(trust_score) as avg_ts FROM sessions WHERE status = 'active'"
        ).fetchone()
        avg_trust = avg_row["avg_ts"] if avg_row["avg_ts"] is not None else 0.0

        risk_counts = {}
        for tier in ["low", "medium", "high"]:
            count = conn.execute(
                "SELECT COUNT(*) as cnt FROM sessions WHERE risk_tier = ? AND status = 'active'",
                (tier,),
            ).fetchone()["cnt"]
            risk_counts[tier] = count

    return DashboardSummary(
        active_sessions=active,
        total_alerts=total_alerts,
        unacknowledged_alerts=unack,
        average_trust_score=round(avg_trust, 4),
        sessions_by_risk=risk_counts,
    )


@router.get("/alerts", response_model=AlertListResponse)
async def list_alerts(limit: int = 50, offset: int = 0):
    """List SOC alerts, most recent first."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM alerts ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        total = conn.execute("SELECT COUNT(*) as cnt FROM alerts").fetchone()["cnt"]

    alerts = [
        AlertResponse(
            id=r["id"],
            session_id=r["session_id"],
            alert_type=AlertType(r["alert_type"]),
            details=None,  # TODO: parse JSON details
            acknowledged=bool(r["acknowledged"]),
            created_at=r["created_at"],
        )
        for r in rows
    ]

    return AlertListResponse(alerts=alerts, total=total)


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: int):
    """Mark an alert as acknowledged."""
    with get_db() as conn:
        result = conn.execute(
            "UPDATE alerts SET acknowledged = 1 WHERE id = ?", (alert_id,)
        )
        if result.rowcount == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Alert not found",
            )
    return {"status": "acknowledged", "alert_id": alert_id}


@router.get("/events/{session_id}", response_model=List[TrustEventResponse])
async def get_trust_events(session_id: str, limit: int = 200):
    """Get trust event timeline for a specific session."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM trust_events WHERE session_id = ? ORDER BY timestamp ASC LIMIT ?",
            (session_id, limit),
        ).fetchall()

    return [
        TrustEventResponse(
            id=r["id"],
            session_id=r["session_id"],
            trust_score=r["trust_score"],
            facial_score=r["facial_score"],
            liveness_score=r["liveness_score"],
            behavioral_score=r["behavioral_score"],
            risk_tier=RiskTier(r["risk_tier"]) if r["risk_tier"] else RiskTier.LOW,
            policy_action=PolicyAction(r["policy_action"]) if r["policy_action"] else PolicyAction.ALLOW,
            shap_attributions=None,  # TODO (Phase 5): parse JSON
            timestamp=r["timestamp"],
        )
        for r in rows
    ]
