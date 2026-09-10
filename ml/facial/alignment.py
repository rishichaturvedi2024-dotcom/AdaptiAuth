"""
AdaptiAuth — ML Layer 1: Face Detection and Alignment
Uses MTCNN from facenet-pytorch to detect faces and extract aligned crops.
"""

import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN

class FaceAligner:
    """
    Handles face detection and alignment using MTCNN.
    Outputs a normalized tensor ready for InceptionResnetV1.
    """
    def __init__(self, device=None, image_size=160, margin=0):
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.image_size = image_size
        
        # Initialize MTCNN
        # keep_all=False ensures we only grab the primary (most prominent) face.
        self.mtcnn = MTCNN(
            image_size=image_size, 
            margin=margin, 
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], 
            factor=0.709, 
            post_process=True, # Normalizes output tensor to [-1, 1]
            device=self.device,
            keep_all=False 
        )

    def align(self, image: np.ndarray) -> torch.Tensor | None:
        """
        Takes an image (BGR or RGB numpy array) and returns the aligned face crop tensor.
        Returns None if no face is detected.
        
        Args:
            image: numpy array of shape (H, W, 3). If loaded via OpenCV, it's typically BGR.
                   We'll convert to RGB for MTCNN.
                   
        Returns:
            Aligned face tensor of shape (3, image_size, image_size) or None.
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Expected numpy array for image input.")
            
        # MTCNN works best with PIL Images in RGB format
        if image.shape[2] == 3:
            # Assuming BGR from OpenCV if the source is webcam; convert to RGB
            # Note: This is a heuristic. In production, ensure inputs are explicitly handled.
            img_rgb = image[..., ::-1].copy() 
        else:
            img_rgb = image
            
        pil_img = Image.fromarray(img_rgb)
        
        # Extract face (returns a tensor because post_process=True)
        # Returns None if no face is found
        face_tensor = self.mtcnn(pil_img)
        
        return face_tensor
