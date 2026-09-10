import pytest
import numpy as np
import torch
from ml.liveness.rppg import RPPGExtractor
from ml.liveness.pad import PADDetector

@pytest.fixture
def rppg():
    return RPPGExtractor(fps=30)

@pytest.fixture
def pad():
    return PADDetector()

def test_rppg_extractor_static_image(rppg):
    # Static image fed repeatedly should yield low variance/liveness
    static_frame = np.ones((300, 300, 3), dtype=np.uint8) * 128
    
    # Feed 31 frames (more than 1 second buffer)
    for _ in range(31):
        score = rppg.process_frame(static_frame)
        
    assert score == 0.0 # Definitely a photo/spoof due to zero variance
    
def test_pad_detector(pad):
    # Dummy face tensor
    dummy_face = torch.randn(3, 160, 160)
    
    live_prob = pad.detect(dummy_face)
    
    assert 0.0 <= live_prob <= 1.0

def test_pad_detector_spoof(pad):
    # Spoof image (uniform color)
    spoof_face = torch.zeros(3, 160, 160)
    
    live_prob = pad.detect(spoof_face)
    
    if not pad._model_loaded:
        pytest.skip("PAD model weights not loaded")
        
    assert 0.0 <= live_prob <= 1.0  # Just verify it returns a valid probability
