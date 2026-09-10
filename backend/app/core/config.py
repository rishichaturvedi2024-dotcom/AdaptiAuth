"""
AdaptiAuth — Application Configuration

Loads settings from environment variables / .env file.
"""

from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Central configuration loaded from .env or environment variables."""

    # ── Application ──────────────────────────────────────────
    app_name: str = "AdaptiAuth"
    app_env: str = "development"
    debug: bool = True
    secret_key: str = "change-me-to-a-random-secret-key-in-production"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ── Database ─────────────────────────────────────────────
    database_url: str = "sqlite:///./adaptiauth.db"

    # ── JWT / Session ────────────────────────────────────────
    jwt_secret_key: str = "change-me-jwt-secret"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30

    # ── Trust Engine ─────────────────────────────────────────
    trust_rescore_interval_seconds: int = 5
    trust_low_threshold: float = 0.80
    trust_medium_threshold: float = 0.50

    # ── WebAuthn ─────────────────────────────────────────────
    webauthn_rp_id: str = "localhost"
    webauthn_rp_name: str = "AdaptiAuth Demo"
    webauthn_origin: str = "http://localhost:5173"

    # ── ML Models ────────────────────────────────────────────
    facial_model_path: str = "ml/facial/weights/"
    liveness_pad_model_path: str = "ml/liveness/weights/"
    behavioral_model_path: str = "ml/behavioral/weights/"

    # ── Frontend / CORS ──────────────────────────────────────
    frontend_url: str = "http://localhost:5173"
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    # ── SOC Webhook ──────────────────────────────────────────
    soc_webhook_url: str = "http://localhost:9999/webhook/soc-alert"

    @property
    def cors_origin_list(self) -> List[str]:
        return [origin.strip() for origin in self.cors_origins.split(",")]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


# Singleton settings instance
settings = Settings()
