"""
AdaptiAuth — ML Layer 1: Facial Verification Service (Backend Binding)

Bridges the FastAPI backend to the ML facial verification pipeline.
"""

from typing import List, Optional, Tuple
import numpy as np
import torch
import sys
import os

# Ensure the root of the project is in path so we can import ml
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from ml.facial.alignment import FaceAligner
from ml.facial.embedding import FaceEmbedder
from ml.facial.matching import compute_similarity, is_match

class FacialVerificationService:
    """
    Service for facial embedding extraction and identity verification.

    Phase 1 Implements:
    - Face detection (MTCNN)
    - 3D face alignment
    - CNN embedding extraction (InceptionResnetV1, 512-dim)
    - Cosine similarity matching against enrolled templates
    """

    def __init__(self):
        self.aligner = FaceAligner()
        self.embedder = FaceEmbedder()

    def extract_embedding(self, frame: np.ndarray) -> Optional[List[float]]:
        """
        Extract a 512-dim facial embedding from an RGB or BGR frame.
        Returns None if no face is detected.
        """
        face_tensor = self.aligner.align(frame)
        if face_tensor is None:
            return None
            
        embedding_tensor = self.embedder.extract_embedding(face_tensor)
        return embedding_tensor.cpu().numpy().tolist()

    def match(
        self, embedding: List[float], template: List[float]
    ) -> Tuple[float, float]:
        """
        Compare an embedding against an enrolled template.
        Returns (similarity_score, confidence).
        """
        tensor1 = torch.tensor(embedding)
        tensor2 = torch.tensor(template)
        
        sim_score = compute_similarity(tensor1, tensor2)
        
        # Confidence logic: Map similarity range to [0, 1] confidence heuristically
        # Using 0.6 as a threshold base
        confidence = max(0.0, min(1.0, (sim_score + 0.2) / 1.2))
        
        return (sim_score, confidence)

# Module-level singleton
facial_service = FacialVerificationService()

