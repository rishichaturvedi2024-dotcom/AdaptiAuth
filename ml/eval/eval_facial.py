"""
AdaptiAuth — Validation & Metrics Scaffold: Layer 1 (Facial Verification)

Evaluates the Layer 1 facial verification pipeline using the Labeled Faces in the Wild (LFW) dataset.

### Setup Instructions
1. Download the LFW images: http://vis-www.cs.umass.edu/lfw/lfw.tgz
2. Download the `pairs.txt` protocol: http://vis-www.cs.umass.edu/lfw/pairs.txt
3. Extract `lfw.tgz` into `data/lfw/` so you have `data/lfw/Aaron_Eckhart/` etc.
4. Place `pairs.txt` in `data/lfw/pairs.txt`.
"""

import os
import sys
import json
import time
import cv2
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.app.services.facial import FacialVerificationService
from tests.evaluation.evaluate_metrics import calculate_far_frr

def load_pairs(pairs_path):
    """
    Parses LFW pairs.txt or pairs.csv.
    Returns matched: list of (name, id1, id2)
            mismatched: list of (name1, id1, name2, id2)
    """
    matched = []
    mismatched = []
    
    is_csv = pairs_path.endswith('.csv')
    
    with open(pairs_path, 'r') as f:
        lines = f.readlines()
        
    # Skip header
    start_idx = 1
    
    for line in lines[start_idx:]:
        line = line.strip()
        if not line: continue
        
        if is_csv:
            # remove trailing comma if present
            if line.endswith(','):
                line = line[:-1]
            parts = line.split(',')
        else:
            parts = line.split()
            
        if len(parts) == 3:
            matched.append((parts[0], int(parts[1]), int(parts[2])))
        elif len(parts) == 4:
            mismatched.append((parts[0], int(parts[1]), parts[2], int(parts[3])))
            
    return matched, mismatched

def get_image_path(lfw_dir, name, img_num):
    filename = f"{name}_{img_num:04d}.jpg"
    return os.path.join(lfw_dir, name, filename)

def run_evaluation(lfw_dir, pairs_path, results_json_path):
    if not os.path.exists(lfw_dir) or not os.path.exists(pairs_path):
        print("ERROR: LFW dataset not found.")
        print("Please follow the setup instructions in the script docstring to download and place it in data/lfw/.")
        return

    print("Initializing FacialVerificationService...")
    facial_service = FacialVerificationService()
    
    matched, mismatched = load_pairs(pairs_path)
    
    # We may not want to run all 6000 pairs if it takes too long on CPU, 
    # but let's run a subset if limit is needed, or just run all.
    # For robust testing, we'll run all or a large chunk.
    print(f"Loaded {len(matched)} matched pairs and {len(mismatched)} mismatched pairs.")
    
    scores = []
    labels = []
    
    total = len(matched) + len(mismatched)
    processed = 0
    failures = 0
    
    start_time = time.time()
    
    # Process matched pairs
    for name, id1, id2 in matched:
        img1_path = get_image_path(os.path.join(lfw_dir, 'lfw-deepfunneled', 'lfw-deepfunneled'), name, id1)
        img2_path = get_image_path(os.path.join(lfw_dir, 'lfw-deepfunneled', 'lfw-deepfunneled'), name, id2)
        
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            failures += 1
            continue
            
        emb1 = facial_service.extract_embedding(img1)
        emb2 = facial_service.extract_embedding(img2)
        
        if emb1 and emb2:
            sim, _ = facial_service.match(emb1, emb2)
            scores.append(sim)
            labels.append(True)
        else:
            failures += 1
            
        processed += 1
        if processed % 100 == 0:
            print(f"Processed {processed}/{total} pairs...")

    # Process mismatched pairs
    for name1, id1, name2, id2 in mismatched:
        img1_path = get_image_path(os.path.join(lfw_dir, 'lfw-deepfunneled', 'lfw-deepfunneled'), name1, id1)
        img2_path = get_image_path(os.path.join(lfw_dir, 'lfw-deepfunneled', 'lfw-deepfunneled'), name2, id2)
        
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            failures += 1
            continue
            
        emb1 = facial_service.extract_embedding(img1)
        emb2 = facial_service.extract_embedding(img2)
        
        if emb1 and emb2:
            sim, _ = facial_service.match(emb1, emb2)
            scores.append(sim)
            labels.append(False)
        else:
            failures += 1
            
        processed += 1
        if processed % 100 == 0:
            print(f"Processed {processed}/{total} pairs...")

    elapsed = time.time() - start_time
    print(f"\nEvaluation completed in {elapsed:.2f}s")
    print(f"Face detection failures (skipped): {failures}")
    
    # Threshold is defined in ml/facial/matching.py as 0.6
    threshold = 0.6
    far, frr = calculate_far_frr(scores, labels, threshold=threshold)
    
    # Compute accuracy
    correct = 0
    for s, l in zip(scores, labels):
        pred = (s >= threshold)
        if pred == l:
            correct += 1
    accuracy = correct / len(scores) if len(scores) > 0 else 0
    
    results = {
        "dataset": "LFW (Labeled Faces in the Wild)",
        "pairs_tested": len(scores),
        "failures": failures,
        "threshold": threshold,
        "accuracy": accuracy,
        "far": far,
        "frr": frr,
        "execution_time_seconds": elapsed
    }
    
    print("\n=== LFW Evaluation Results ===")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"FAR: {far*100:.2f}%")
    print(f"FRR: {frr*100:.2f}%")
    
    # Update results JSON
    if os.path.exists(results_json_path):
        with open(results_json_path, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = {}
        
    if "facial" not in all_results:
        all_results["facial"] = {}
        
    all_results["facial"]["LFW"] = results
    
    with open(results_json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved results to {results_json_path}")

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    lfw_dir = os.path.join(base_dir, 'data', 'lfw')
    
    # In some LFW distributions it's pairs.txt, in ours it's pairs.csv or matchpairsDevTest.csv
    # Let's use pairs.csv
    pairs_path = os.path.join(lfw_dir, 'pairs.csv')
    
    # If pairs.csv doesn't exist but pairs.txt does, use that
    if not os.path.exists(pairs_path) and os.path.exists(os.path.join(lfw_dir, 'pairs.txt')):
        pairs_path = os.path.join(lfw_dir, 'pairs.txt')
        
    results_json = os.path.join(base_dir, 'docs', 'validation_results.json')
    
    run_evaluation(lfw_dir, pairs_path, results_json)
