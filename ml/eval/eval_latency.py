"""
AdaptiAuth — Validation & Metrics Scaffold: End-to-End Latency

Evaluates the true end-to-end latency of the full trust-fusion pipeline 
(Facial + PAD + rPPG + Behavioral -> TrustEngine + SHAP).
"""

import os
import sys
import json
import time
import cv2
import statistics

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ml.facial.alignment import FaceAligner
from backend.app.services.facial import FacialVerificationService
from ml.liveness.pad import PADDetector
from ml.liveness.rppg import RPPGExtractor
from ml.behavioral.sequence import BehavioralModel
from ml.fusion.trust_engine import TrustEngine
from ml.explainability.shap_explainer import TrustSHAPExplainer

def get_sample_image(base_dir):
    # Grab a sample image from LFW
    lfw_dir = os.path.join(base_dir, 'data', 'lfw', 'lfw-deepfunneled', 'lfw-deepfunneled')
    for dp, dn, fn in os.walk(lfw_dir):
        for f in fn:
            if f.endswith('.jpg'):
                return cv2.imread(os.path.join(dp, f))
    return None

def run_evaluation(results_json_path, base_dir):
    print("Initializing Full Pipeline Components...")
    
    # 1. Initialize models
    aligner = FaceAligner()
    facial_svc = FacialVerificationService()
    pad = PADDetector()
    rppg = RPPGExtractor(fps=30)
    beh_model = BehavioralModel()
    
    engine = TrustEngine()
    explainer = TrustSHAPExplainer(engine)
    
    # Pre-compute an embedding to compare against
    sample_img = get_sample_image(base_dir)
    if sample_img is None:
        print("ERROR: No sample image found in LFW for latency testing.")
        return
        
    reference_emb = facial_svc.extract_embedding(sample_img)
    if reference_emb is None:
        print("ERROR: Failed to extract reference embedding.")
        return
    
    sample_payload = {
        "session_id": "latency_test",
        "keystrokes": [{"key": "x", "timestamp": i*100} for i in range(11)],
        "mouse_events": [],
        "device_fingerprint": "eval_device",
        "timestamp": time.time()
    }
    
    print("Warming up models (10 runs)...")
    for _ in range(10):
        # Full stack warmup
        emb = facial_svc.extract_embedding(sample_img)
        if emb:
            s, c = facial_svc.match(emb, reference_emb)
        face_tensor = aligner.align(sample_img)
        if face_tensor is not None:
            pad.detect(face_tensor)
        rppg.process_frame(sample_img)
        beh_model.evaluate(sample_payload)
        engine.compute_trust_score(0.9, 0.9, 0.8, 0.8, 0.7, 0.7)
        explainer.explain(0.9, 0.9, 0.8, 0.8, 0.7, 0.7)
        
    runs = 20
    print(f"Timing over {runs} full-stack end-to-end runs...")
    
    latencies = []
    
    for i in range(runs):
        start_time = time.perf_counter()
        
        # 1. Vision Layer (Facial + PAD + rPPG)
        emb = facial_svc.extract_embedding(sample_img)
        face_tensor = aligner.align(sample_img)
        if emb is not None and face_tensor is not None:
            facial_s, facial_c = facial_svc.match(emb, reference_emb)
            pad_s = pad.detect(face_tensor)
            pad_c = 0.8 # stub confidence
        else:
            facial_s, facial_c, pad_s, pad_c = 0.0, 0.0, 0.0, 0.0
            
        rppg_s = rppg.process_frame(sample_img)
        rppg_c = 0.7
        
        live_s = (pad_s + rppg_s) / 2.0
        live_c = (pad_c + rppg_c) / 2.0
        
        # 2. Behavioral Layer
        beh_res = beh_model.evaluate(sample_payload)
        beh_s = (beh_res["keystroke_score"] * 0.7) + (beh_res["context_score"] * 0.3)
        beh_c = 0.85
        
        # 3. Trust Engine & SHAP
        score_res = engine.compute_trust_score(
            facial_s, facial_c, live_s, live_c, beh_s, beh_c
        )
        
        explanations = explainer.explain(
            facial_s, facial_c, live_s, live_c, beh_s, beh_c
        )
        
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000) # ms
        
    mean_ms = statistics.mean(latencies)
    median_ms = statistics.median(latencies)
    min_ms = min(latencies)
    max_ms = max(latencies)
    std_dev_ms = statistics.stdev(latencies) if len(latencies) > 1 else 0
    
    print("\n=== End-to-End Latency Results ===")
    print(f"Runs: {runs}")
    print(f"Mean Latency:   {mean_ms:.2f} ms")
    print(f"Median Latency: {median_ms:.2f} ms")
    print(f"Min Latency:    {min_ms:.2f} ms")
    print(f"Max Latency:    {max_ms:.2f} ms")
    print(f"Std Deviation:  {std_dev_ms:.2f} ms")
    
    results = {
        "status": "measured",
        "phase": 10,
        "runs": runs,
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "std_dev_ms": std_dev_ms,
        "notes": "True end-to-end latency (image processing -> face alignment -> embeddings -> pad_cnn -> rppg_dsp -> behavioral_svm -> trust fusion -> SHAP)"
    }
    
    if os.path.exists(results_json_path):
        with open(results_json_path, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = {}
        
    all_results["latency_phase10"] = results
    
    with open(results_json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved Phase 10 results to {results_json_path}")

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    results_json = os.path.join(base_dir, 'docs', 'validation_results.json')
    run_evaluation(results_json, base_dir)
