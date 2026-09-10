import pytest
from ml.fusion.trust_engine import TrustEngine
from ml.explainability.shap_explainer import TrustSHAPExplainer

@pytest.fixture(scope="module")
def trust_engine():
    return TrustEngine()

@pytest.fixture(scope="module")
def shap_explainer(trust_engine):
    return TrustSHAPExplainer(trust_engine)

def test_trust_engine_weights(trust_engine):
    # Test equal confidence
    res = trust_engine.compute_trust_score(
        facial_score=1.0, facial_conf=1.0,
        liveness_score=1.0, liveness_conf=1.0,
        behavioral_score=1.0, behavioral_conf=1.0
    )
    assert res["trust_score"] == 1.0
    assert res["weights"]["facial"] == 0.40
    
    # Test zero confidence
    res2 = trust_engine.compute_trust_score(
        facial_score=1.0, facial_conf=0.0,
        liveness_score=1.0, liveness_conf=0.0,
        behavioral_score=1.0, behavioral_conf=0.0
    )
    assert res2["weights"]["facial"] == 0.40 # Fallback

def test_shap_explainer(shap_explainer):
    attributions = shap_explainer.explain(
        facial_score=0.9, facial_conf=0.9,
        liveness_score=0.1, liveness_conf=0.9,
        behavioral_score=0.8, behavioral_conf=0.9
    )
    
    assert len(attributions) == 6
    # liveness is 0.1, which is low, should have a negative attribution
    liveness_attr = next(a for a in attributions if a["feature"] == "liveness_score")
    assert liveness_attr["direction"] == "negative"
    
    # facial is 0.9, high, should have positive attribution
    facial_attr = next(a for a in attributions if a["feature"] == "facial_score")
    assert facial_attr["direction"] == "positive"
