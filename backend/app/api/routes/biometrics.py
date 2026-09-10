"""
AdaptiAuth — Biometrics Routes

Endpoints for processing raw biometric signals (face, liveness, behavior)
and returning the layer-specific scores/embeddings.
"""

from typing import List, Optional
import io
import numpy as np
from PIL import Image

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status

from app.models.schemas import (
    FacialVerificationResult,
    LivenessResult,
    BehavioralPayload,
    BehavioralResult,
)

# Initialize ML models (lazy load or global, we'll keep it global for the prototype)
import sys
import os
# We need to make sure 'ml' module can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))

from ml.facial.alignment import FaceAligner
from ml.facial.embedding import FaceEmbedder
from ml.facial.matching import compute_similarity, is_match

from ml.liveness.rppg import RPPGExtractor
from ml.liveness.pad import PADDetector
from ml.behavioral.sequence import BehavioralModel

router = APIRouter(prefix="/biometrics", tags=["Biometrics"])

face_aligner = FaceAligner()
face_embedder = FaceEmbedder()

pad_detector = PADDetector()
# We would typically need one rPPG extractor per session, but we'll use a global one for the prototype
rppg_extractor = RPPGExtractor()

behavioral_model = BehavioralModel()

# For prototype purposes, we keep a dummy "enrolled" template in memory
# In a real app, this would be fetched from the database for the current user
_dummy_template = None


@router.post("/face", response_model=FacialVerificationResult)
async def verify_face(
    file: UploadFile = File(...),
    enroll: bool = Form(False, description="Set to true to enroll this face as the template")
):
    """
    Process an image frame, extract a facial embedding, and compute similarity
    against the enrolled template.
    """
    global _dummy_template
    
    # Read image bytes
    contents = await file.read()
    
    # Convert to numpy array
    try:
        # Load image via PIL to ensure standard RGB processing
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    # Step 1: Align face
    face_tensor = face_aligner.align(img_np)
    if face_tensor is None:
        raise HTTPException(status_code=400, detail="No face detected in the image")
        
    # Step 2: Extract embedding
    embedding_tensor = face_embedder.extract_embedding(face_tensor)
    embedding_list = embedding_tensor.tolist()
    
    # Handle enrollment
    if enroll:
        _dummy_template = embedding_tensor
        return FacialVerificationResult(
            embedding=embedding_list,
            similarity_score=1.0,
            confidence=1.0,
            aligned=True
        )
        
    # Step 3: Match against template
    if _dummy_template is None:
        # If no template is enrolled, we'll just return a similarity of 0
        similarity_score = 0.0
    else:
        similarity_score = compute_similarity(embedding_tensor, _dummy_template)
        
    # Heuristic confidence (how clear is the face? just a stub for now)
    confidence = 0.95
    
    # Normalize similarity score to 0-1 for the trust score (cosine sim is -1 to 1)
    normalized_sim = max(0.0, (similarity_score + 1) / 2)

    return FacialVerificationResult(
        embedding=embedding_list,
        similarity_score=normalized_sim,
        confidence=confidence,
        aligned=True
    )


@router.post("/liveness", response_model=LivenessResult)
async def check_liveness(
    file: UploadFile = File(...)
):
    """
    Process an image frame for liveness detection using rPPG and PAD.
    """
    contents = await file.read()
    
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    # 1. Align face for PAD (Needs 160x160 tensor)
    face_tensor = face_aligner.align(img_np)
    
    if face_tensor is None:
        raise HTTPException(status_code=400, detail="No face detected in the image")
        
    # 2. Presentation Attack Detection (PAD)
    # PAD returns "liveness probability" (1.0 - spoof_prob)
    pad_live_prob = pad_detector.detect(face_tensor)
    
    # 3. rPPG extraction (needs the original frame for ROI cropping)
    rppg_conf = rppg_extractor.process_frame(img_np)
    
    # 4. Simple fusion for liveness sub-score
    # Real logic would be weighted: e.g. 0.7 * PAD + 0.3 * rPPG
    # For prototype:
    fused_liveness = 0.6 * pad_live_prob + 0.4 * rppg_conf
    
    return LivenessResult(
        rppg_confidence=rppg_conf,
        pad_spoof_probability=(1.0 - pad_live_prob),
        liveness_score=fused_liveness,
        is_live=(fused_liveness > 0.6)
    )

@router.post("/behavior", response_model=BehavioralResult)
async def analyze_behavior(payload: BehavioralPayload):
    """
    Process behavioral events (keystrokes, mouse) to generate a consistency score.
    """
    # Convert Pydantic model to dict for the ML layer
    payload_dict = payload.dict()
    
    # Evaluate sequence
    scores = behavioral_model.evaluate(payload_dict)
    
    # Simple fusion for behavioral consistency
    key_s = scores["keystroke_score"]
    mouse_s = scores["mouse_score"]
    ctx_s = scores["context_score"]
    
    fused_behavior = (key_s * 0.4) + (mouse_s * 0.4) + (ctx_s * 0.2)
    
    return BehavioralResult(
        keystroke_score=key_s,
        mouse_score=mouse_s,
        context_score=ctx_s,
        behavioral_consistency=fused_behavior,
        confidence=scores["confidence"]
    )
