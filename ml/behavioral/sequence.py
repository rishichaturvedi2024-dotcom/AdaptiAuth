"""
AdaptiAuth — ML Layer 3: Behavioral Sequence Model
Scores the consistency of keystrokes, mouse movement, and context.
Upgraded to OneClassSVM for Keystroke Dynamics (Phase 10).
"""

from typing import List, Dict, Any
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

class BehavioralModel:
    """
    Evaluates behavioral payloads against a session baseline using One-Class SVM.
    """
    def __init__(self):
        # Maps session_id or user_id to their trained model and scaler
        self.profiles = {}
        
    def enroll(self, session_id: str, feature_vectors: List[List[float]]):
        """
        Train a behavioral profile (OneClassSVM) for a user based on enrollment samples.
        """
        if not feature_vectors:
            return
            
        X = np.array(feature_vectors)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # nu=0.05 allows ~5% of training samples to be considered outliers (margin)
        model = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
        model.fit(X_scaled)
        
        self.profiles[session_id] = {
            "model": model,
            "scaler": scaler
        }
        
    def evaluate(self, payload: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate a sequence of behavior (keystrokes, mouse, context).
        Returns a dictionary of scores.
        """
        session_id = payload.get("session_id", "unknown")
        features = payload.get("features", [])
        if not features:
            features = self._extract_features(payload)
        
        # If no profile exists, or no features are passed, return default/fallback
        if session_id not in self.profiles or not features:
            return {
                "keystroke_score": self._keystroke_score(payload),
                "mouse_score": self._mouse_score(payload),
                "context_score": 0.99 if payload.get("device_fingerprint") else 0.5,
                "confidence": min(0.85, 0.25 + min(len(features) / 20.0, 0.6))
            }
            
        profile = self.profiles[session_id]
        model = profile["model"]
        scaler = profile["scaler"]
        
        X = np.array(features).reshape(1, -1)
        X_scaled = scaler.transform(X)
        
        # decision_function returns positive for inliers and negative for outliers
        raw_score = model.decision_function(X_scaled)[0]
        
        # Map raw score to a 0.0 - 1.0 confidence score
        # Multiply by a scaling factor to stretch the curve reasonably.
        import math
        mapped_score = 1 / (1 + math.exp(-raw_score * 2.0))
        
        # Context score
        context_score = 0.99 if payload.get("device_fingerprint") else 0.5
        
        return {
            "keystroke_score": mapped_score,
            "mouse_score": self._mouse_score(payload),
            "context_score": context_score,
            "confidence": 0.9
        }

    @staticmethod
    def _extract_features(payload: Dict[str, Any]) -> List[float]:
        keys = payload.get("keystrokes", [])
        moves = payload.get("mouse_events", [])
        key_times = [float(event["timestamp"]) for event in keys if "timestamp" in event]
        move_times = [float(event["timestamp"]) for event in moves if "timestamp" in event]
        key_intervals = np.diff(key_times) if len(key_times) > 1 else np.array([0.0])
        move_intervals = np.diff(move_times) if len(move_times) > 1 else np.array([0.0])
        return [
            float(len(keys)),
            float(np.mean(key_intervals)),
            float(np.std(key_intervals)),
            float(len(moves)),
            float(np.mean(move_intervals)),
            float(np.std(move_intervals)),
        ]

    @staticmethod
    def _keystroke_score(payload: Dict[str, Any]) -> float:
        events = payload.get("keystrokes", [])
        if len(events) < 2:
            return 0.5
        timestamps = np.array([float(event["timestamp"]) for event in events])
        intervals = np.diff(timestamps)
        valid = intervals[(intervals > 0) & (intervals < 3000)]
        if len(valid) == 0:
            return 0.35
        variability = float(np.std(valid) / (np.mean(valid) + 1e-6))
        return float(np.clip(0.9 - variability * 0.35, 0.35, 0.95))

    @staticmethod
    def _mouse_score(payload: Dict[str, Any]) -> float:
        events = payload.get("mouse_events", [])
        if len(events) < 2:
            return 0.8
        distances = []
        for previous, current in zip(events, events[1:]):
            dx = float(current.get("x", 0)) - float(previous.get("x", 0))
            dy = float(current.get("y", 0)) - float(previous.get("y", 0))
            distances.append(float(np.hypot(dx, dy)))
        if not distances:
            return 0.5
        movement = float(np.mean(distances))
        return float(np.clip(0.45 + min(movement / 500.0, 0.45), 0.45, 0.9))
