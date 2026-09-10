"""
AdaptiAuth — ML Layer 1: Face Matching
Computes cosine similarity between facial embeddings.
"""

import torch
import torch.nn.functional as F

def compute_similarity(embedding1: torch.Tensor, embedding2: torch.Tensor) -> float:
    """
    Computes the cosine similarity between two embedding vectors.
    
    Args:
        embedding1: 1D Tensor (e.g., 512 dimensions)
        embedding2: 1D Tensor (e.g., 512 dimensions)
        
    Returns:
        float: Similarity score between -1.0 and 1.0. 
               1.0 means identical, 0.0 means orthogonal, -1.0 means opposite.
               Typically, for facenet models, a score > 0.6 indicates a match.
    """
    # Ensure they are 1D
    e1 = embedding1.squeeze()
    e2 = embedding2.squeeze()
    
    if e1.dim() != 1 or e2.dim() != 1:
        raise ValueError("Embeddings must be 1D tensors.")
        
    # Compute cosine similarity: (A dot B) / (||A|| * ||B||)
    # F.cosine_similarity expects 2D tensors (batch_size, features)
    cos_sim = F.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0))
    
    return cos_sim.item()

def is_match(similarity_score: float, threshold: float = 0.6) -> bool:
    """
    Determines if two embeddings belong to the same person based on a threshold.
    """
    return similarity_score >= threshold
