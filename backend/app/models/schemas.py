"""
AdaptiAuth — Pydantic Schemas

Request/response models for the API, and internal data transfer objects.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ─── Enums ───────────────────────────────────────────────────

class RiskTier(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SessionStatus(str, Enum):
    ACTIVE = "active"
    STEP_UP = "step_up"
    TERMINATED = "terminated"


class PolicyAction(str, Enum):
    ALLOW = "allow"
    STEP_UP = "step_up"
    TERMINATE = "terminate"


class AlertType(str, Enum):
    STEP_UP_REQUIRED = "step_up_required"
    SESSION_TERMINATED = "session_terminated"


# ─── Auth ────────────────────────────────────────────────────

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)


class UserResponse(BaseModel):
    id: str
    username: str
    created_at: datetime


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    session_id: str
    user_id: str


# ─── Biometric Signals ──────────────────────────────────────

class FacialVerificationResult(BaseModel):
    """Output from Layer 1: Deep Facial Verification."""
    embedding: List[float] = Field(..., description="128-dim facial embedding")
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    aligned: bool = True


class LivenessResult(BaseModel):
    """Output from Layer 2: Dual-Layer Liveness Detection."""
    rppg_confidence: float = Field(..., ge=0.0, le=1.0, description="Physiological liveness confidence from rPPG")
    pad_spoof_probability: float = Field(..., ge=0.0, le=1.0, description="Presentation-attack spoof probability")
    liveness_score: float = Field(..., ge=0.0, le=1.0, description="Fused liveness sub-score")
    is_live: bool


class BehavioralResult(BaseModel):
    """Output from Layer 3: Behavioral & Contextual Monitoring."""
    keystroke_score: float = Field(..., ge=0.0, le=1.0)
    mouse_score: float = Field(..., ge=0.0, le=1.0)
    context_score: float = Field(..., ge=0.0, le=1.0)
    behavioral_consistency: float = Field(..., ge=0.0, le=1.0, description="Combined behavioral score")
    confidence: float = Field(..., ge=0.0, le=1.0)


# ─── Trust Score ─────────────────────────────────────────────

class TrustScoreResult(BaseModel):
    """Output from the Adaptive Trust-Scoring Engine."""
    trust_score: float = Field(..., ge=0.0, le=1.0)
    facial_score: Optional[float] = None
    liveness_score: Optional[float] = None
    behavioral_score: Optional[float] = None
    weights: Dict[str, float] = Field(default_factory=dict, description="Weights used in fusion")
    risk_tier: RiskTier
    policy_action: PolicyAction
    timestamp: datetime


class SHAPAttribution(BaseModel):
    """A single SHAP feature attribution."""
    feature: str
    value: float = Field(description="Input feature value")
    attribution: float = Field(description="SHAP attribution (contribution to trust score)")
    direction: str = Field(description="'positive' or 'negative' contribution")


class ExplainedTrustScore(BaseModel):
    """Trust score with full SHAP explainability payload."""
    trust_score: float
    risk_tier: RiskTier
    policy_action: PolicyAction
    attributions: List[SHAPAttribution]
    base_value: float = Field(description="SHAP base/expected value")
    timestamp: datetime


# ─── Session ─────────────────────────────────────────────────

class SessionResponse(BaseModel):
    id: str
    user_id: str
    trust_score: float
    risk_tier: RiskTier
    status: SessionStatus
    created_at: datetime
    updated_at: datetime


class SessionListResponse(BaseModel):
    sessions: List[SessionResponse]
    total: int


# ─── SOC Alerts ──────────────────────────────────────────────

class AlertResponse(BaseModel):
    id: int
    session_id: str
    alert_type: AlertType
    details: Optional[Dict[str, Any]] = None
    acknowledged: bool
    created_at: datetime


class AlertListResponse(BaseModel):
    alerts: List[AlertResponse]
    total: int


# ─── Dashboard ───────────────────────────────────────────────

class TrustEventResponse(BaseModel):
    """A single trust-score event for the timeline chart."""
    id: int
    session_id: str
    trust_score: float
    facial_score: Optional[float]
    liveness_score: Optional[float]
    behavioral_score: Optional[float]
    risk_tier: RiskTier
    policy_action: PolicyAction
    shap_attributions: Optional[List[SHAPAttribution]] = None
    timestamp: datetime


class DashboardSummary(BaseModel):
    """Summary stats for the SOC dashboard."""
    active_sessions: int
    total_alerts: int
    unacknowledged_alerts: int
    average_trust_score: float
    sessions_by_risk: Dict[str, int]


# ─── Behavioral capture (from frontend) ─────────────────────

class KeystrokeEvent(BaseModel):
    key: str
    event_type: str = Field(description="'keydown' or 'keyup'")
    timestamp: float = Field(description="Unix timestamp in ms")


class MouseEvent(BaseModel):
    x: float
    y: float
    event_type: str = Field(description="'mousemove', 'click', etc.")
    timestamp: float


class BehavioralPayload(BaseModel):
    """Batch of behavioral events from the frontend."""
    session_id: str
    keystrokes: List[KeystrokeEvent] = Field(default_factory=list)
    mouse_events: List[MouseEvent] = Field(default_factory=list)
    device_fingerprint: Optional[str] = None
    ip_address: Optional[str] = None
    timestamp: float


# ─── Health Check ────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    app_name: str
    environment: str
    version: str = "0.1.0"
