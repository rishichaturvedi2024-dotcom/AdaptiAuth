"""
AdaptiAuth — Zero-Trust Policy Engine
Maps Trust Scores to risk tiers and actions.
"""

from app.models.schemas import RiskTier, PolicyAction

class PolicyEngine:
    def __init__(self):
        # Configurable thresholds
        self.threshold_low = 0.80
        self.threshold_medium = 0.50
        
    def evaluate(self, trust_score: float) -> dict:
        """
        Evaluate a trust score and return the corresponding risk tier and action.
        
        Rules:
        - 0.80 - 1.00 : Low Risk -> Uninterrupted access (ALLOW)
        - 0.50 - 0.79 : Medium Risk -> WebAuthn step-up (STEP_UP)
        - 0.00 - 0.49 : High Risk -> Terminate session + SOC alert (TERMINATE)
        """
        if trust_score >= self.threshold_low:
            return {
                "risk_tier": RiskTier.LOW,
                "policy_action": PolicyAction.ALLOW
            }
        elif trust_score >= self.threshold_medium:
            return {
                "risk_tier": RiskTier.MEDIUM,
                "policy_action": PolicyAction.STEP_UP
            }
        else:
            return {
                "risk_tier": RiskTier.HIGH,
                "policy_action": PolicyAction.TERMINATE
            }

    def execute_action(self, session_id: str, action: PolicyAction):
        """
        Execute the specific action for a session.
        In the prototype, we mock these side effects.
        """
        if action == PolicyAction.ALLOW:
            # Do nothing, session continues
            print(f"[PolicyEngine] Session {session_id}: ALLOW")
            pass
        elif action == PolicyAction.STEP_UP:
            # Trigger WebAuthn flow (update DB session status to 'step_up')
            print(f"[PolicyEngine] Session {session_id}: STEP_UP required")
            self._trigger_webauthn(session_id)
        elif action == PolicyAction.TERMINATE:
            # Terminate session and raise SOC alert
            print(f"[PolicyEngine] Session {session_id}: TERMINATE")
            self._terminate_session(session_id)
            self._raise_soc_alert(session_id)
            
    def _trigger_webauthn(self, session_id: str):
        # Stub: update DB state so frontend knows to challenge
        pass
        
    def _terminate_session(self, session_id: str):
        # Stub: update DB state to 'terminated' so token becomes invalid
        pass
        
    def _raise_soc_alert(self, session_id: str):
        # Stub: write an alert to the SOC alerts table
        pass
