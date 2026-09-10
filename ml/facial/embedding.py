"""
AdaptiAuth — ML Layer 1: Facial Embedding
Uses InceptionResnetV1 from facenet-pytorch to extract 512-d facial embeddings.
"""

import torch
from facenet_pytorch import InceptionResnetV1

class FaceEmbedder:
    """
    Extracts high-dimensional feature embeddings from aligned face crops.
    Returns 512-dimensional vectors.
    """
    def __init__(self, device=None, pretrained='vggface2'):
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        # Initialize ResNet
        # Set to eval mode as we are only using it for inference
        self.resnet = InceptionResnetV1(pretrained=pretrained).eval().to(self.device)

    def extract_embedding(self, face_tensor: torch.Tensor) -> torch.Tensor:
        """
        Takes an aligned face tensor and returns its embedding vector.
        
        Args:
            face_tensor: Tensor of shape (3, 160, 160) or (1, 3, 160, 160).
                         Values should be normalized to [-1, 1].
                         
        Returns:
            1D Tensor representing the 512-dim embedding (if input is single image),
            or 2D Tensor (B, 512) if batched.
        """
        if face_tensor.dim() == 3:
            # Add batch dimension
            face_tensor = face_tensor.unsqueeze(0)
            
        face_tensor = face_tensor.to(self.device)
        
        with torch.no_grad():
            embeddings = self.resnet(face_tensor)
            
        # If input was 3D (single image), return 1D tensor
        if face_tensor.size(0) == 1:
            return embeddings.squeeze(0)
            
        return embeddings
