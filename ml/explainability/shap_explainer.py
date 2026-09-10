"""
AdaptiAuth — ML Layer 5: Explainability
Uses SHAP to explain the Trust Engine's dynamically weighted score.
"""

import shap
import numpy as np
from typing import List, Dict, Any

class TrustSHAPExplainer:
    def __init__(self, trust_engine):
        self.trust_engine = trust_engine
        
        # Base/Expected value for Trust Score
        # For prototype, we'll assume a baseline score of 0.5 when all inputs are 0.5
        self.base_value = 0.5
        
        # We wrap the trust engine's compute method so SHAP can call it
        # SHAP expects a function f(X) where X is a matrix of features (num_samples, num_features)
        # Features: [facial_score, facial_conf, liveness_score, liveness_conf, behavioral_score, behavioral_conf]
        def predict_fn(X):
            scores = []
            for row in X:
                res = self.trust_engine.compute_trust_score(row[0], row[1], row[2], row[3], row[4], row[5])
                scores.append(res["trust_score"])
            return np.array(scores)
            
        self.predict_fn = predict_fn
        
        # Initialize an ExactExplainer or KernelExplainer
        # We use a dummy background dataset of "neutral" 0.5 scores
        background = np.array([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])
        self.explainer = shap.KernelExplainer(self.predict_fn, background)
        
    def explain(
        self, 
        facial_score: float, facial_conf: float,
        liveness_score: float, liveness_conf: float,
        behavioral_score: float, behavioral_conf: float
    ) -> List[Dict[str, Any]]:
        """
        Explain a single trust score computation.
        Returns a list of attribution dictionaries.
        """
        # SHAP expects 2D array
        X = np.array([[facial_score, facial_conf, liveness_score, liveness_conf, behavioral_score, behavioral_conf]])
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X, silent=True)
        
        # shap_values could be a list for multi-class, but here it's an array for regression
        if isinstance(shap_values, list):
            vals = shap_values[0][0]
        else:
            vals = shap_values[0]
            
        feature_names = [
            "facial_score", "facial_conf", 
            "liveness_score", "liveness_conf", 
            "behavioral_score", "behavioral_conf"
        ]
        
        attributions = []
        for i, name in enumerate(feature_names):
            attr_val = float(vals[i])
            direction = "positive" if attr_val >= 0 else "negative"
            attributions.append({
                "feature": name,
                "value": float(X[0][i]),
                "attribution": abs(attr_val),
                "direction": direction
            })
            
        # Sort by impact
        attributions.sort(key=lambda x: x["attribution"], reverse=True)
        return attributions
