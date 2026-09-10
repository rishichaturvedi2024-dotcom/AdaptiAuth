"""
AdaptiAuth — Database Layer

SQLite-backed storage for the prototype.
Swap-in note: Replace the DATABASE_URL with a PostgreSQL connection string
and change `create_engine` accordingly for production use.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.core.config import settings


# ── Database path ────────────────────────────────────────────
# Extract path from sqlite:/// URL
_db_path = settings.database_url.replace("sqlite:///", "")
DB_PATH = Path(_db_path).resolve()


def get_connection() -> sqlite3.Connection:
    """Create a new database connection with row_factory enabled."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def get_db():
    """Context manager for database connections."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Initialize the database schema. Safe to call multiple times (IF NOT EXISTS)."""
    with get_db() as conn:
        conn.executescript("""
            -- User enrollment
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                facial_template BLOB,
                behavioral_baseline BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Active sessions
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT REFERENCES users(id),
                trust_score REAL DEFAULT 1.0,
                risk_tier TEXT DEFAULT 'low',
                status TEXT DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Trust score history (for dashboard timeline)
            CREATE TABLE IF NOT EXISTS trust_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT REFERENCES sessions(id),
                trust_score REAL NOT NULL,
                facial_score REAL,
                liveness_score REAL,
                behavioral_score REAL,
                shap_attributions TEXT,
                risk_tier TEXT,
                policy_action TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- SOC alerts
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT REFERENCES sessions(id),
                alert_type TEXT NOT NULL,
                details TEXT,
                acknowledged INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- WebAuthn credentials (for step-up)
            CREATE TABLE IF NOT EXISTS webauthn_credentials (
                id TEXT PRIMARY KEY,
                user_id TEXT REFERENCES users(id),
                credential_id BLOB NOT NULL,
                public_key BLOB NOT NULL,
                sign_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
