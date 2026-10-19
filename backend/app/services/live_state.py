"""In-memory live signal state for the prototype session monitor."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Dict, Optional


@dataclass
class SessionSignals:
    facial_score: float = 0.0
    facial_confidence: float = 0.0
    liveness_score: float = 0.0
    liveness_confidence: float = 0.0
    behavioral_score: float = 0.5
    behavioral_confidence: float = 0.0
    keystroke_score: float = 0.5
    mouse_score: float = 0.5
    context_score: float = 0.5
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    face_updated_at: Optional[datetime] = None


class LiveSignalStore:
    def __init__(self) -> None:
        self._signals: Dict[str, SessionSignals] = {}
        self._lock = Lock()

    def get(self, session_id: str) -> SessionSignals:
        with self._lock:
            return self._signals.setdefault(session_id, SessionSignals())

    def update(self, session_id: str, **values: float) -> SessionSignals:
        with self._lock:
            signals = self._signals.setdefault(session_id, SessionSignals())
            for key, value in values.items():
                if hasattr(signals, key) and value is not None:
                    setattr(signals, key, max(0.0, min(1.0, float(value))))
            if any(key in values for key in ("facial_score", "liveness_score")):
                signals.face_updated_at = datetime.now(timezone.utc)
            signals.updated_at = datetime.now(timezone.utc)
            return signals

    def mark_face_absent(self, session_id: str) -> SessionSignals:
        return self.update(
            session_id,
            facial_score=0.0,
            facial_confidence=1.0,
            liveness_score=0.0,
            liveness_confidence=1.0,
        )

    def mark_stale_faces_absent(self, session_id: str, max_age_seconds: float = 6.0) -> SessionSignals:
        signals = self.get(session_id)
        if (
            signals.face_updated_at is None
            or datetime.now(timezone.utc) - signals.face_updated_at
            > timedelta(seconds=max_age_seconds)
        ):
            return self.mark_face_absent(session_id)
        return signals


live_signal_store = LiveSignalStore()
