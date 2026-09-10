import pytest
import torch
import numpy as np
from PIL import Image

from ml.facial.alignment import FaceAligner
from ml.facial.embedding import FaceEmbedder
from ml.facial.matching import compute_similarity, is_match

@pytest.fixture(scope="module")
def aligner():
    # Use CPU for tests to ensure they run anywhere
    return FaceAligner(device=torch.device('cpu'))

@pytest.fixture(scope="module")
def embedder():
    return FaceEmbedder(device=torch.device('cpu'))

def test_face_alignment(aligner):
    # Create a dummy image that looks like a face (or just random noise for MTCNN to fail/pass)
    # Since MTCNN actually needs to detect a face, providing noise will return None
    dummy_img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    
    tensor = aligner.align(dummy_img)
    # Because it's random noise, it shouldn't find a face
    assert tensor is None

def test_face_embedding(embedder):
    # Dummy aligned face tensor (3, 160, 160)
    dummy_face = torch.randn(3, 160, 160)
    
    embedding = embedder.extract_embedding(dummy_face)
    
    assert embedding is not None
    assert embedding.dim() == 1
    assert embedding.size(0) == 512

def test_face_matching():
    # Create two dummy embeddings
    emb1 = torch.randn(512)
    emb2 = torch.randn(512)
    
    sim = compute_similarity(emb1, emb2)
    assert -1.0 <= sim <= 1.0
    
    # Same embedding should match perfectly
    sim_same = compute_similarity(emb1, emb1)
    assert pytest.approx(sim_same, 0.01) == 1.0
    assert is_match(sim_same, threshold=0.6)
