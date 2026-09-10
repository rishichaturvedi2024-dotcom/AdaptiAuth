"""
AdaptiAuth — Authentication Routes

Handles user registration, login, and initial session creation.
"""

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, status

from app.core.database import get_db
from app.core.security import hash_password, verify_password, create_access_token
from app.models.schemas import (
    UserCreate,
    UserResponse,
    LoginRequest,
    LoginResponse,
)

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user: UserCreate):
    """Register a new user."""
    user_id = str(uuid.uuid4())
    password_hash = hash_password(user.password)

    with get_db() as conn:
        # Check uniqueness
        existing = conn.execute(
            "SELECT id FROM users WHERE username = ?", (user.username,)
        ).fetchone()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Username already exists",
            )

        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO users (id, username, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (user_id, user.username, password_hash, now),
        )

    return UserResponse(id=user_id, username=user.username, created_at=now)


@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """Authenticate and create a session."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT id, username, password_hash FROM users WHERE username = ?",
            (request.username,),
        ).fetchone()

    if not row or not verify_password(request.password, row["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    user_id = row["id"]
    session_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    # Create session record
    with get_db() as conn:
        conn.execute(
            "INSERT INTO sessions (id, user_id, trust_score, risk_tier, status, created_at, updated_at) "
            "VALUES (?, ?, 1.0, 'low', 'active', ?, ?)",
            (session_id, user_id, now, now),
        )

    # Issue JWT
    access_token = create_access_token(
        data={"sub": user_id, "session_id": session_id}
    )

    return LoginResponse(
        access_token=access_token,
        session_id=session_id,
        user_id=user_id,
    )
