"""
AdaptiAuth — ML Layer 4: Adaptive Trust-Scoring Engine
Dynamically weights and fuses signals from the 3 lower layers into a single Trust Score.
"""

from typing import Dict, Any

class TrustEngine:
    """
    Fuses biometric scores to produce a single continuous trust score.
    """
    def __init__(self):
        # Base weights when all confidence levels are equal
        self.base_weights = {
            "facial": 0.40,
            "liveness": 0.35,
            "behavioral": 0.25
        }
        
    def compute_trust_score(
        self, 
        facial_score: float, facial_conf: float,
        liveness_score: float, liveness_conf: float,
        behavioral_score: float, behavioral_conf: float
    ) -> Dict[str, Any]:
        """
        Computes the final trust score, dynamically adjusting weights
        based on the confidence of each signal layer.
        """
        # Dynamic weighting based on confidence
        raw_w_f = self.base_weights["facial"] * facial_conf
        raw_w_l = self.base_weights["liveness"] * liveness_conf
        raw_w_b = self.base_weights["behavioral"] * behavioral_conf
        
        total_w = raw_w_f + raw_w_l + raw_w_b
        
        # Handle edge case where all confidences are zero (or total is 0)
        if total_w == 0:
            w_f, w_l, w_b = self.base_weights["facial"], self.base_weights["liveness"], self.base_weights["behavioral"]
        else:
            w_f = raw_w_f / total_w
            w_l = raw_w_l / total_w
            w_b = raw_w_b / total_w
            
        trust_score = (w_f * facial_score) + (w_l * liveness_score) + (w_b * behavioral_score)
        
        return {
            "trust_score": float(trust_score),
            "weights": {
                "facial": float(w_f),
                "liveness": float(w_l),
                "behavioral": float(w_b)
            }
        }
