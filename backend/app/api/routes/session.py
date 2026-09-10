"""
AdaptiAuth — Session Management Routes

Session queries, status updates, and trust-score history retrieval.
Endpoints will be enriched in later phases as the trust engine is built.
"""

from fastapi import APIRouter, HTTPException, status

from app.core.database import get_db
from app.models.schemas import (
    SessionResponse,
    SessionListResponse,
    RiskTier,
    SessionStatus,
)

router = APIRouter(prefix="/sessions", tags=["Sessions"])


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    """Retrieve a session by ID."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()

    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    return SessionResponse(
        id=row["id"],
        user_id=row["user_id"],
        trust_score=row["trust_score"],
        risk_tier=RiskTier(row["risk_tier"]),
        status=SessionStatus(row["status"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


@router.get("/", response_model=SessionListResponse)
async def list_sessions(status_filter: str = None, limit: int = 50, offset: int = 0):
    """List sessions, optionally filtering by status."""
    with get_db() as conn:
        if status_filter:
            rows = conn.execute(
                "SELECT * FROM sessions WHERE status = ? ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                (status_filter, limit, offset),
            ).fetchall()
            total_row = conn.execute(
                "SELECT COUNT(*) as cnt FROM sessions WHERE status = ?",
                (status_filter,),
            ).fetchone()
        else:
            rows = conn.execute(
                "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
            total_row = conn.execute(
                "SELECT COUNT(*) as cnt FROM sessions"
            ).fetchone()

    sessions = [
        SessionResponse(
            id=r["id"],
            user_id=r["user_id"],
            trust_score=r["trust_score"],
            risk_tier=RiskTier(r["risk_tier"]),
            status=SessionStatus(r["status"]),
            created_at=r["created_at"],
            updated_at=r["updated_at"],
        )
        for r in rows
    ]

    return SessionListResponse(sessions=sessions, total=total_row["cnt"])
