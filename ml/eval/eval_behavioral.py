"""
AdaptiAuth — Validation & Metrics Scaffold: Layer 3 (Behavioral Biometrics)

Evaluates the Layer 3 behavioral sequence model using the CMU Keystroke Dynamics Benchmark Dataset.
Phase 10 Upgraded: Genuine evaluation of OneClassSVM.
"""

import os
import sys
import json
import time
import csv
import random

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ml.behavioral.sequence import BehavioralModel

def load_cmu_dataset(csv_path):
    """
    Loads CMU keystroke dynamics dataset.
    Returns a dict mapping subjects to their lists of feature vectors.
    """
    data_by_subject = {}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            subject = row[0]
            # Features start from index 3
            features = [float(x) for x in row[3:]]
            if subject not in data_by_subject:
                data_by_subject[subject] = []
            data_by_subject[subject].append(features)
            
    return data_by_subject

def generate_payload(session_id, features):
    return {
        "session_id": session_id,
        "features": features,
        "device_fingerprint": "eval_device",
        "timestamp": time.time()
    }

def run_evaluation(csv_path, results_json_path):
    if not os.path.exists(csv_path):
        print("ERROR: CMU Keystroke dataset not found.")
        return

    print("Initializing BehavioralModel (Phase 10 OneClassSVM)...")
    model = BehavioralModel()
    
    print("Loading dataset...")
    data_by_subject = load_cmu_dataset(csv_path)
    subjects = list(data_by_subject.keys())
    print(f"Loaded {len(subjects)} subjects.")
    
    scores = []
    labels = []
    
    start_time = time.time()
    
    # We will enroll each subject using their first 200 samples.
    # Then we test on the remaining 200 samples as genuine.
    # For impostors, we randomly select 200 samples from other subjects.
    
    for subject in subjects:
        samples = data_by_subject[subject]
        
        enrollment_samples = samples[:200]
        genuine_test_samples = samples[200:400]
        
        # Pick impostor samples randomly from other subjects
        other_subjects = [s for s in subjects if s != subject]
        impostor_test_samples = []
        for _ in range(200):
            imp_sub = random.choice(other_subjects)
            impostor_test_samples.append(random.choice(data_by_subject[imp_sub]))
            
        # Enroll
        model.enroll(subject, enrollment_samples)
        
        # Test Genuine
        for sample in genuine_test_samples:
            payload = generate_payload(subject, sample)
            res = model.evaluate(payload)
            scores.append(res["keystroke_score"])
            labels.append(True)
            
        # Test Impostor
        for sample in impostor_test_samples:
            payload = generate_payload(subject, sample)
            res = model.evaluate(payload)
            scores.append(res["keystroke_score"])
            labels.append(False)

    elapsed = time.time() - start_time
    print(f"\nEvaluation completed in {elapsed:.2f}s")
    
    threshold = 0.5 # Sigmoid mapping makes 0 -> 0.5
    correct = 0
    false_accepts = 0
    false_rejects = 0
    genuines = 0
    impostors = 0
    
    for s, l in zip(scores, labels):
        pred = (s >= threshold)
        if pred == l:
            correct += 1
            
        if l:
            genuines += 1
            if not pred: false_rejects += 1
        else:
            impostors += 1
            if pred: false_accepts += 1
            
    accuracy = correct / len(scores) if len(scores) > 0 else 0
    far = false_accepts / impostors if impostors > 0 else 0
    frr = false_rejects / genuines if genuines > 0 else 0
    
    tp = genuines - false_rejects
    tn = impostors - false_accepts
    fp = false_accepts
    fn = false_rejects
    
    results = {
        "status": "measured",
        "phase": 10,
        "model": "Scikit-Learn OneClassSVM (RBF Kernel)",
        "dataset": "CMU Keystroke Dynamics Benchmark",
        "protocol": "Enrollment on first 200 samples. Tested on remaining 200 genuine + 200 random impostors per subject.",
        "samples_tested": len(scores),
        "genuine_samples": genuines,
        "impostor_samples": impostors,
        "threshold": threshold,
        "accuracy": accuracy,
        "far": far,
        "frr": frr,
        "precision": tp / (tp + fp) if tp + fp > 0 else 0,
        "recall": tp / (tp + fn) if tp + fn > 0 else 0,
        "confusion_matrix": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        "execution_time_seconds": elapsed
    }
    
    print("\n=== Behavioral Evaluation Results ===")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"FAR: {far*100:.2f}%")
    print(f"FRR: {frr*100:.2f}%")
    
    if os.path.exists(results_json_path):
        with open(results_json_path, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = {}
        
    all_results["behavioral_phase10"] = results
    
    with open(results_json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved results to {results_json_path}")

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    csv_path = os.path.join(base_dir, 'data', 'cmu_keystroke', 'DSL-StrongPasswordData.csv')
    results_json = os.path.join(base_dir, 'docs', 'validation_results.json')
    
    run_evaluation(csv_path, results_json)
