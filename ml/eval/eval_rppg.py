"""
AdaptiAuth — Validation & Metrics Scaffold: Layer 2 (rPPG / Physiological Liveness)

Evaluates the Layer 2 rPPG extraction pipeline on:
1. NFI rPPG-Deepfake
2. Kaggle rPPG dataset
"""

import os
import sys
import json
import time
import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ml.liveness.rppg import RPPGExtractor
from tests.evaluation.evaluate_metrics import calculate_far_frr

def process_video(video_path, max_frames=150):
    """
    Extracts rPPG confidence. We cap at max_frames (e.g. 5 seconds at 30fps)
    to keep evaluation time reasonable for large videos.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return 0.5
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30
    
    extractor = RPPGExtractor(fps=int(fps))
    final_conf = 0.5
    frames_read = 0
    
    while cap.isOpened() and frames_read < max_frames:
        ret, frame = cap.read()
        if not ret: break
        
        final_conf = extractor.process_frame(frame)
        frames_read += 1
        
    cap.release()
    return final_conf

def evaluate_nfi(base_dir):
    print("\n--- Evaluating NFI rPPG-Deepfake Dataset ---")
    nfi_dir = os.path.join(base_dir, 'data', 'rppg_nfi')
    if not os.path.exists(nfi_dir):
        return {"status": "blocked", "reason": f"Directory not found: {nfi_dir}"}
        
    live_paths = [os.path.join(dp, f) for dp, dn, filenames in os.walk(os.path.join(nfi_dir, 'real')) for f in filenames if f.lower().endswith(('.mp4','.avi','.mov'))]
    spoof_paths = [os.path.join(dp, f) for dp, dn, filenames in os.walk(os.path.join(nfi_dir, 'fake')) for f in filenames if f.lower().endswith(('.mp4','.avi','.mov'))]
    
    print(f"NFI: Found {len(live_paths)} real, {len(spoof_paths)} fake.")
    return _run_rppg_evaluation(live_paths, spoof_paths)

def evaluate_kaggle(base_dir):
    print("\n--- Evaluating Kaggle rPPG Dataset ---")
    kaggle_dir = os.path.join(base_dir, 'data', 'rppg_kaggle')
    if not os.path.exists(kaggle_dir):
        return {"status": "blocked", "reason": f"Directory not found: {kaggle_dir}"}
        
    # The Kaggle dataset contains only genuine physiological recordings.
    # There are no 'fake'/'spoof' labels.
    live_paths = [os.path.join(dp, f) for dp, dn, filenames in os.walk(kaggle_dir) for f in filenames if f.lower().endswith(('.mp4','.avi','.mov','.mov'))]
    
    print(f"Kaggle rPPG: Found {len(live_paths)} genuine videos.")
    
    if not live_paths:
        return {"status": "blocked", "reason": "No videos found in Kaggle rPPG."}
        
    start_time = time.time()
    scores = []
    
    for idx, p in enumerate(live_paths):
        scores.append(process_video(p))
        if (idx+1) % 10 == 0:
            print(f"  Processed {idx+1}/{len(live_paths)} Kaggle videos...")
            
    elapsed = time.time() - start_time
    
    threshold = 0.5
    frr = sum(1 for s in scores if s < threshold) / len(scores) if len(scores) > 0 else 0
    
    return {
        "status": "measured",
        "phase": 10,
        "model": "DSP (Butterworth Bandpass + FFT SNR)",
        "sample_count": len(scores),
        "threshold": threshold,
        "mean_live_confidence": float(np.mean(scores)),
        "median_live_confidence": float(np.median(scores)),
        "frr": frr,
        "execution_time_seconds": elapsed,
        "notes": "Dataset only contains genuine physiological recordings. Measured FRR."
    }

def _run_rppg_evaluation(live_paths, spoof_paths):
    start_time = time.time()
    scores, labels = [], []
    
    # Optional cap to speed up tests during dev, but user requested FULL evaluation
    # We will run all samples but cap each video to 5 seconds.
    
    for idx, p in enumerate(live_paths):
        scores.append(process_video(p))
        labels.append(True)
        if (idx+1) % 10 == 0: print(f"  Processed {idx+1}/{len(live_paths)} live videos...")
        
    for idx, p in enumerate(spoof_paths):
        scores.append(process_video(p))
        labels.append(False)
        if (idx+1) % 10 == 0: print(f"  Processed {idx+1}/{len(spoof_paths)} spoof videos...")
        
    if not scores:
        return {"status": "blocked", "reason": "No valid videos found."}
        
    live_scores = [s for s, l in zip(scores, labels) if l]
    spoof_scores = [s for s, l in zip(scores, labels) if not l]
    
    threshold = 0.5
    far, frr = calculate_far_frr(scores, labels, threshold=threshold)
    accuracy = sum((s >= threshold) == l for s, l in zip(scores, labels)) / len(scores)
    
    return {
        "status": "measured",
        "phase": 10,
        "model": "DSP (Butterworth Bandpass + FFT SNR)",
        "sample_count": len(scores),
        "threshold": threshold,
        "mean_live_confidence": float(np.mean(live_scores)) if live_scores else 0.0,
        "median_live_confidence": float(np.median(live_scores)) if live_scores else 0.0,
        "mean_spoof_confidence": float(np.mean(spoof_scores)) if spoof_scores else 0.0,
        "median_spoof_confidence": float(np.median(spoof_scores)) if spoof_scores else 0.0,
        "accuracy": accuracy,
        "far": far,
        "frr": frr,
        "execution_time_seconds": time.time() - start_time
    }

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    results_json = os.path.join(base_dir, 'docs', 'validation_results.json')
    
    res_nfi = evaluate_nfi(base_dir)
    res_kaggle = evaluate_kaggle(base_dir)
    
    if os.path.exists(results_json):
        with open(results_json, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = {}
        
    if "liveness_rppg_phase10" not in all_results:
        all_results["liveness_rppg_phase10"] = {}
        
    all_results["liveness_rppg_phase10"]["NFI-rPPG-Deepfake"] = res_nfi
    all_results["liveness_rppg_phase10"]["Kaggle-rPPG"] = res_kaggle
    
    with open(results_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved rPPG Phase 10 results to {results_json}")
