import pytest
from ml.behavioral.sequence import BehavioralModel

@pytest.fixture
def behavioral_model():
    return BehavioralModel()

def test_behavioral_model_fallback(behavioral_model):
    payload = {
        "session_id": "test_1",
        "device_fingerprint": "abc"
    }
    scores = behavioral_model.evaluate(payload)
    
    assert scores["keystroke_score"] == 0.5
    assert scores["mouse_score"] == 0.8
    assert scores["context_score"] == 0.99

def test_behavioral_model_enrolled(behavioral_model):
    # Dummy features (e.g. key timing vectors)
    enroll_features = [[0.1, 0.2, 0.3], [0.12, 0.21, 0.29], [0.09, 0.19, 0.31], [0.1, 0.22, 0.3]]
    behavioral_model.enroll("test_2", enroll_features)
    
    payload = {
        "session_id": "test_2",
        "features": [0.1, 0.2, 0.3],
        "device_fingerprint": "abc"
    }
    scores = behavioral_model.evaluate(payload)
    
    assert scores["keystroke_score"] > 0.49
    assert scores["mouse_score"] == 0.8
    assert scores["context_score"] == 0.99

def test_behavioral_model_anomaly(behavioral_model):
    enroll_features = [[0.1, 0.2, 0.3], [0.12, 0.21, 0.29], [0.09, 0.19, 0.31], [0.1, 0.22, 0.3]]
    behavioral_model.enroll("test_3", enroll_features)
    
    # Anomaly vector far from training distribution
    payload = {
        "session_id": "test_3",
        "features": [9.0, 9.0, 9.0],
        "device_fingerprint": "abc"
    }
    scores = behavioral_model.evaluate(payload)
    
    assert scores["keystroke_score"] < 0.5
