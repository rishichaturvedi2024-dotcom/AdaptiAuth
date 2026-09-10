"""
AdaptiAuth — ML Layer 2: Presentation Attack Detection (PAD)
Evaluates whether a presented face is a live person or a spoof.
Phase 10 Upgraded: Genuine PyTorch CNN.

CALIBRATION NOTE (Phase 10 Live Audit):
  The PAD CNN was trained on full uncropped NUAA images (face + background).
  Live inference receives MTCNN-cropped face-only tensors.
  This module handles both input domains:
    - Raw numpy frames (full scene) → training-identical preprocessing
    - MTCNN face tensors [-1,1] → rescaled to [0,1] + ImageNet normalize
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Explicit PAD result states
PAD_VALID = "VALID"
PAD_UNAVAILABLE = "UNAVAILABLE"
PAD_FAILED = "FAILED"

class SimplePADCNN(nn.Module):
    """
    Extremely lightweight CNN for CPU-friendly inference (< 50ms).
    Input: 128x128 RGB
    """
    def __init__(self):
        super(SimplePADCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 64x64
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 32x32
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 16x16
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class PADDetector:
    """
    Detects presentation attacks using a trained PyTorch CNN.
    
    Supports two input modes:
      1. detect(face_tensor) — MTCNN face crop tensor (shape 3,H,W or 1,3,H,W)
      2. detect_frame(frame) — Raw BGR numpy frame (full scene, matches training domain)
    
    The CNN was trained on full uncropped NUAA images. For best accuracy,
    use detect_frame() with the full webcam frame. detect() on MTCNN crops
    works but operates on an out-of-distribution input domain.
    """
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SimplePADCNN().to(self.device)
        
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        model_path = os.path.join(base_dir, 'data', 'models', 'pad_cnn.pth')
        self._debug_dir = os.path.join(base_dir, 'debug')
        self._debug_saved = False  # Only save debug crop once
        
        # Load weights if they exist, else initialize randomly (and warn)
        self._model_loaded = False
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            self._model_loaded = True
        else:
            print(f"WARNING: PAD model weights not found at {model_path}. Using random initialization!")
            
        self.model.eval()
        
        # Transform for MTCNN face crops (already [0,1] after rescaling)
        self._crop_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Transform for full-frame images — matches training exactly
        # PIL Image → [0,1] float tensor → ImageNet normalize
        self._frame_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def detect_frame_with_diagnostics(self, frame: np.ndarray) -> dict:
        """
        Evaluate PAD on a full BGR frame using TRAINING-IDENTICAL preprocessing.
        Returns a dictionary with full diagnostic information.
        """
        if frame is None or frame.size == 0:
            return {"score": 0.0, "state": PAD_UNAVAILABLE, "logit": 0.0}
            
        if not self._model_loaded:
            return {"score": 0.0, "state": PAD_UNAVAILABLE, "logit": 0.0}
        
        try:
            # Convert BGR -> RGB -> PIL (matches training pipeline exactly)
            rgb = frame[..., ::-1].copy()
            pil_img = Image.fromarray(rgb)
            
            # Hook into the pre-sigmoid linear layer to get raw logits
            pre_sigmoid_val = None
            def hook_fn(module, input, output):
                nonlocal pre_sigmoid_val
                pre_sigmoid_val = output.item()
                
            hook = self.model.classifier[2].register_forward_hook(hook_fn)
            
            with torch.no_grad():
                tensor = self._frame_transform(pil_img).unsqueeze(0).to(self.device)
                output = self.model(tensor)
                score = output.item()
                
            hook.remove()
                
            # Save exact input frame once
            if not self._debug_saved:
                import cv2
                os.makedirs(self._debug_dir, exist_ok=True)
                cv2.imwrite(os.path.join(self._debug_dir, "live_pad_input.jpg"), frame)
                self._save_debug_crop(tensor.squeeze(0), "live_pad_tensor.jpg")
                self._debug_saved = True
                
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
            checkpoint_path = os.path.join(base_dir, 'data', 'models', 'pad_cnn.pth')
                
            return {
                "score": score,
                "logit": pre_sigmoid_val,
                "state": PAD_VALID,
                "frame_shape": frame.shape,
                "tensor_shape": tuple(tensor.shape),
                "preprocess_fn": "BGR -> RGB -> PIL -> Resize(128) -> ToTensor -> Normalize",
                "class_mapping": "1.0=LIVE, 0.0=SPOOF",
                "checkpoint": checkpoint_path
            }
            
        except Exception as e:
            print(f"PAD detect_frame error: {e}")
            return {"score": 0.0, "state": PAD_FAILED, "logit": 0.0}

    def detect(self, face_tensor: torch.Tensor) -> float:
        """
        Evaluate PAD confidence on an MTCNN face crop tensor.
        
        NOTE: The CNN was trained on full uncropped images, not face crops.
        This path applies correct normalization but operates on a different
        input domain than training. Use detect_frame() for best accuracy.
        
        Input: face_tensor of shape (3, H, W) or (1, 3, H, W).
        Returns: confidence score [0.0, 1.0], where 1.0 is live, 0.0 is spoof.
        """
        if face_tensor is None:
            return 0.0
            
        with torch.no_grad():
            if face_tensor.dim() == 3:
                face_tensor = face_tensor.unsqueeze(0) # (1, 3, H, W)
                
            # If the tensor is from MTCNN [-1, 1], map it back to [0, 1] for standard PyTorch normalization
            if face_tensor.min() < 0:
                face_tensor = (face_tensor + 1.0) / 2.0
                
            face_tensor = face_tensor.to(self.device)
            face_tensor = self._crop_transform(face_tensor)
            
            # Save debug crop once
            if not self._debug_saved:
                self._save_debug_crop(face_tensor.squeeze(0), "live_pad_crop.jpg")
            
            output = self.model(face_tensor)
            score = output.item()
            
        return score

    def _save_debug_crop(self, tensor, filename):
        """Save the exact tensor being fed into the CNN as a debug image (once)."""
        try:
            os.makedirs(self._debug_dir, exist_ok=True)
            # Undo ImageNet normalization for visualization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            t = tensor.cpu().clone()
            t = t * std + mean
            t = t.clamp(0, 1)
            img = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            import cv2
            cv2.imwrite(os.path.join(self._debug_dir, filename), img[:, :, ::-1])
            self._debug_saved = True
            print(f"[PAD Debug] Saved input crop: debug/{filename}")
        except Exception as e:
            print(f"[PAD Debug] Could not save crop: {e}")
