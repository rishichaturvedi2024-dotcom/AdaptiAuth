import pytest
from ml.behavioral.sequence import BehavioralModel

@pytest.fixture
def behavioral_model():
    return BehavioralModel()

def test_behavioral_model_empty(behavioral_model):
    payload = {
        "session_id": "test_1",
        "keystrokes": [],
        "mouse_events": []
    }
    scores = behavioral_model.evaluate(payload)
    
    assert scores["keystroke_score"] == 0.8
    assert scores["mouse_score"] == 0.8
    assert scores["context_score"] == 0.5 # Missing device fingerprint

def test_behavioral_model_normal(behavioral_model):
    payload = {
        "session_id": "test_2",
        "keystrokes": [{"key": "a", "event_type": "keydown", "timestamp": 1}],
        "mouse_events": [{"x": 100, "y": 100, "event_type": "mousemove", "timestamp": 1}],
        "device_fingerprint": "abc"
    }
    scores = behavioral_model.evaluate(payload)
    
    assert scores["keystroke_score"] == 0.9
    assert scores["mouse_score"] == 0.95
    assert scores["context_score"] == 0.99

def test_behavioral_model_bot(behavioral_model):
    payload = {
        "session_id": "test_3",
        "keystrokes": [{"key": "a", "event_type": "keydown", "timestamp": i} for i in range(101)],
        "mouse_events": [{"x": i, "y": i, "event_type": "mousemove", "timestamp": i} for i in range(101)],
        "device_fingerprint": "abc"
    }
    scores = behavioral_model.evaluate(payload)
    
    assert scores["keystroke_score"] == 0.5 # Too fast
    assert scores["mouse_score"] == 0.5 # Erratic
