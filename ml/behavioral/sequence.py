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
        
        # If no profile exists, or no features are passed, return default/fallback
        if session_id not in self.profiles or not features:
            return {
                "keystroke_score": 0.5,
                "mouse_score": 0.8,
                "context_score": 0.99 if payload.get("device_fingerprint") else 0.5,
                "confidence": 0.5
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
            "mouse_score": 0.8, # stubbed
            "context_score": context_score,
            "confidence": 0.9
        }
