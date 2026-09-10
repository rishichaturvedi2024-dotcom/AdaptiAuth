"""
AdaptiAuth — Phase 0 Scaffold Tests

Verifies that the FastAPI application skeleton boots correctly,
routes are registered, and the database initializes.
"""

import pytest
from fastapi.testclient import TestClient

# Ensure the backend app module is importable
import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from app.main import app
from app.core.config import settings
from app.core.database import init_db


@pytest.fixture(scope="module")
def client():
    """Create a test client for the FastAPI app."""
    init_db()
    with TestClient(app) as c:
        yield c


class TestHealthAndRoot:
    """Test basic application endpoints."""

    def test_root(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["app"] == "AdaptiAuth"
        assert "docs" in data

    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["app_name"] == "AdaptiAuth"

    def test_docs_available(self, client):
        resp = client.get("/docs")
        assert resp.status_code == 200


class TestAuthRoutes:
    """Test authentication route registration and basic flow."""

    def test_register(self, client):
        import uuid
        username = f"testuser_{uuid.uuid4().hex}"
        resp = client.post("/api/auth/register", json={
            "username": username,
            "password": "securepassword123",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["username"] == username
        assert "id" in data

    def test_register_duplicate(self, client):
        import uuid
        username = f"dupuser_{uuid.uuid4().hex}"
        # First registration
        client.post("/api/auth/register", json={
            "username": username,
            "password": "securepassword123",
        })
        # Second should fail
        resp = client.post("/api/auth/register", json={
            "username": username,
            "password": "securepassword123",
        })
        assert resp.status_code == 409

    def test_login_success(self, client):
        import uuid
        username = f"loginuser_{uuid.uuid4().hex}"
        # Register first
        client.post("/api/auth/register", json={
            "username": username,
            "password": "securepassword123",
        })
        # Login
        resp = client.post("/api/auth/login", json={
            "username": username,
            "password": "securepassword123",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert "session_id" in data
        assert data["token_type"] == "bearer"

    def test_login_wrong_password(self, client):
        import uuid
        username = f"loginuser_{uuid.uuid4().hex}"
        client.post("/api/auth/register", json={
            "username": username,
            "password": "securepassword123",
        })
        resp = client.post("/api/auth/login", json={
            "username": username,
            "password": "wrongpassword",
        })
        assert resp.status_code == 401


class TestSessionRoutes:
    """Test session management route registration."""

    def test_list_sessions(self, client):
        resp = client.get("/api/sessions/")
        assert resp.status_code == 200
        data = resp.json()
        assert "sessions" in data
        assert "total" in data

    def test_get_nonexistent_session(self, client):
        resp = client.get("/api/sessions/nonexistent-id")
        assert resp.status_code == 404


class TestTrustRoutes:
    """Test trust score route registration (stubs)."""

    def test_get_trust_score(self, client):
        resp = client.get("/api/trust/test-session/score")
        assert resp.status_code == 200
        data = resp.json()
        assert "trust_score" in data
        assert "risk_tier" in data
        assert "policy_action" in data

    def test_get_trust_explanation(self, client):
        resp = client.get("/api/trust/test-session/explain")
        assert resp.status_code == 200
        data = resp.json()
        assert "attributions" in data
        assert "base_value" in data
        assert len(data["attributions"]) > 0


class TestDashboardRoutes:
    """Test SOC dashboard route registration."""

    def test_dashboard_summary(self, client):
        resp = client.get("/api/dashboard/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert "active_sessions" in data
        assert "sessions_by_risk" in data

    def test_list_alerts(self, client):
        resp = client.get("/api/dashboard/alerts")
        assert resp.status_code == 200
        data = resp.json()
        assert "alerts" in data
        assert "total" in data


class TestTrustEngine:
    """Test the trust engine's risk classification logic."""

    def test_risk_classification(self):
        from app.services.trust_engine import trust_engine
        from app.models.schemas import RiskTier, PolicyAction

        # Low risk
        tier, action = trust_engine.classify_risk(0.90)
        assert tier == RiskTier.LOW
        assert action == PolicyAction.ALLOW

        # Medium risk
        tier, action = trust_engine.classify_risk(0.65)
        assert tier == RiskTier.MEDIUM
        assert action == PolicyAction.STEP_UP

        # High risk
        tier, action = trust_engine.classify_risk(0.30)
        assert tier == RiskTier.HIGH
        assert action == PolicyAction.TERMINATE

        # Boundary: exactly 0.80
        tier, action = trust_engine.classify_risk(0.80)
        assert tier == RiskTier.LOW

        # Boundary: exactly 0.50
        tier, action = trust_engine.classify_risk(0.50)
        assert tier == RiskTier.MEDIUM

    def test_compute_trust_score(self):
        from app.services.trust_engine import trust_engine

        score, weights = trust_engine.compute_trust_score(
            facial_score=0.90,
            liveness_score=0.85,
            behavioral_score=0.80,
        )
        assert 0.0 <= score <= 1.0
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_missing_signal_handling(self):
        from app.services.trust_engine import trust_engine

        # One missing signal
        score, weights = trust_engine.compute_trust_score(
            facial_score=0.90,
            liveness_score=None,
            behavioral_score=0.80,
        )
        assert 0.0 <= score <= 1.0

        # All missing
        score, _ = trust_engine.compute_trust_score(
            facial_score=None,
            liveness_score=None,
            behavioral_score=None,
        )
        assert score == 0.0
