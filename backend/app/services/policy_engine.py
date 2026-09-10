"""
AdaptiAuth — Zero-Trust Policy Engine Service

Maps Trust Score to a risk tier and executes the corresponding
automated response (allow, step-up, terminate).
Stub implementation until Phase 6.
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Optional

from app.core.database import get_db
from app.models.schemas import AlertType, PolicyAction, RiskTier


class PolicyEngine:
    """
    Zero-Trust Policy Engine — deterministic state machine.

    Risk-tier table:
    ┌──────────────┬────────┬──────────────────────────────────────────┐
    │ Trust Score   │ Risk   │ Response                                 │
    ├──────────────┼────────┼──────────────────────────────────────────┤
    │ 0.80 – 1.00  │ Low    │ Uninterrupted access                     │
    │ 0.50 – 0.79  │ Medium │ FIDO2 / WebAuthn step-up authentication  │
    │ 0.00 – 0.49  │ High   │ Terminate session + generate SOC alert   │
    └──────────────┴────────┴──────────────────────────────────────────┘

    Phase 6 will implement:
    - Full WebAuthn step-up challenge/response flow
    - Session termination with cleanup
    - SOC alert generation with webhook dispatch
    - State transition logging and audit trail
    """

    def execute_policy(
        self,
        session_id: str,
        trust_score: float,
        risk_tier: RiskTier,
        policy_action: PolicyAction,
    ) -> dict:
        """
        Execute the policy action for a session based on its trust score.
        Returns a status dict describing what was done.

        TODO (Phase 6): Wire WebAuthn step-up and SOC webhook.
        """
        result = {
            "session_id": session_id,
            "trust_score": trust_score,
            "risk_tier": risk_tier.value,
            "action": policy_action.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if policy_action == PolicyAction.ALLOW:
            result["detail"] = "Session continues with uninterrupted access."

        elif policy_action == PolicyAction.STEP_UP:
            result["detail"] = "Step-up authentication required (WebAuthn)."
            # TODO (Phase 6): Trigger WebAuthn challenge
            self._update_session_status(session_id, "step_up", risk_tier.value)

        elif policy_action == PolicyAction.TERMINATE:
            result["detail"] = "Session terminated. SOC alert generated."
            self._update_session_status(session_id, "terminated", risk_tier.value)
            self._create_alert(session_id, trust_score)
            # TODO (Phase 6): Fire SOC webhook

        return result

    def _update_session_status(
        self, session_id: str, status: str, risk_tier: str
    ):
        """Update session status in the database."""
        now = datetime.now(timezone.utc).isoformat()
        with get_db() as conn:
            conn.execute(
                "UPDATE sessions SET status = ?, risk_tier = ?, updated_at = ? WHERE id = ?",
                (status, risk_tier, now, session_id),
            )

    def _create_alert(self, session_id: str, trust_score: float):
        """Create a SOC alert for a high-risk session."""
        details = json.dumps({
            "trust_score": trust_score,
            "reason": "Trust score dropped below high-risk threshold",
        })
        with get_db() as conn:
            conn.execute(
                "INSERT INTO alerts (session_id, alert_type, details) VALUES (?, ?, ?)",
                (session_id, AlertType.SESSION_TERMINATED.value, details),
            )


# Module-level singleton
policy_engine = PolicyEngine()
