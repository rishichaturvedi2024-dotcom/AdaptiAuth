import cv2
import numpy as np
import time
import os
import sys
import json
from collections import deque
import torch
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ml.facial.alignment import FaceAligner
from ml.facial.embedding import FaceEmbedder
from ml.facial.matching import compute_similarity
from ml.liveness.pad import PADDetector, PAD_VALID, PAD_UNAVAILABLE
from ml.liveness.rppg import RPPGExtractor
from ml.behavioral.sequence import BehavioralModel
from ml.fusion.trust_engine import TrustEngine
from ml.live.keystroke_collector import KeystrokeCollector

# ---------------------------------------------------------------------------
# Availability states for each modality
# ---------------------------------------------------------------------------
MODALITY_VALID = "VALID"
MODALITY_UNAVAILABLE = "UNAVAILABLE"
MODALITY_FAILED = "FAILED"


class LiveDemoApp:
    def __init__(self):
        print("Initializing Machine Learning Models (This may take a moment)...")
        self.aligner = FaceAligner()
        self.face_auth = FaceEmbedder()
        self.pad = PADDetector()
        self.rppg = RPPGExtractor()
        self.behavioral = BehavioralModel()
        self.trust_engine = TrustEngine()
        self.keys_collector = KeystrokeCollector(target_sequence="start\n")

        self.rppg_buffer = deque(maxlen=150)

        # State variables
        self.enrolled_face_embedding = None
        self.enrolled_behavioral_samples = []  # Stores real enrollment feature vectors
        self.enrollment_sample_count = 0       # Track real enrollment samples
        self.is_enrolled = False
        self.enrollment_mode = False
        self.pending_kb_vec = None

        self.current_face_score = 0.0
        self.current_pad_score = 0.0
        self.current_rppg_score = 0.0
        self.current_behavioral_score = 0.0
        self.current_trust_score = 0.0

        # Explicit modality availability
        self.pad_availability = MODALITY_UNAVAILABLE
        self.rppg_availability = MODALITY_UNAVAILABLE
        self.behavioral_availability = MODALITY_UNAVAILABLE

        self.rppg_state = "Collecting physiological signal..."
        self.face_state = "No Face"
        self.pad_state = "Unknown"
        self.behavioral_state = "Awaiting typing..."
        self.trust_decision = "UNKNOWN"
        self.latency_ms = 0.0
        self.scenario_counter = 0
        self.last_typing_info = ""   # Debug display for typing capture

        # Diagnostic logging — print once per second, not every frame
        self._last_diag_time = 0.0
        self._diag_logged_auth = False  # Log full auth diagnostics once per typing event

        self.cap = cv2.VideoCapture(0)

        # UI Config
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        self.results_path = os.path.join(self.base_dir, 'docs', 'live_demo_results.json')

    def enroll_user(self, frame, typing_samples):
        """
        Live Demo Enrollment.

        Args:
            frame: current BGR webcam frame for face enrollment
            typing_samples: list of feature vectors (each a list of floats).
                            Can be a single sample or multiple real samples.
        """
        print("--- Live Demo Enrollment ---")
        face = self.aligner.align(frame)
        if face is None:
            print("Failed to enroll: No face detected.")
            return False

        self.enrolled_face_embedding = self.face_auth.extract_embedding(face)

        if typing_samples:
            n_real = len(typing_samples)
            print(f"Received {n_real} REAL typing sample(s) for behavioral enrollment.")

            # Log real enrollment samples
            for i, vec in enumerate(typing_samples):
                arr = np.array(vec)
                print(f"  Enrollment sample {i}: shape={arr.shape} mean={arr.mean():.4f} "
                      f"std={arr.std():.4f} min={arr.min():.4f} max={arr.max():.4f}")

            # Bootstrap to 200 samples for OneClassSVM
            # CALIBRATION: σ=0.08 matches natural human typing variability
            # (σ=0.01 was too tight, creating a degenerate decision boundary)
            augmented = []
            for base in typing_samples:
                base_vec = np.array(base)
                samples_per_real = max(1, 200 // n_real)
                for _ in range(samples_per_real):
                    noise = np.random.normal(0, 0.08, len(base_vec))
                    augmented.append((base_vec + noise).tolist())

            # Pad to at least 200
            while len(augmented) < 200:
                base_vec = np.array(typing_samples[np.random.randint(n_real)])
                noise = np.random.normal(0, 0.08, len(base_vec))
                augmented.append((base_vec + noise).tolist())

            self.enrollment_sample_count = n_real
            print(f"Bootstrapped {len(augmented)} training samples from {n_real} real "
                  f"sample(s) with σ=0.08 noise augmentation.")

            self.behavioral.enroll("live_demo_start_user", augmented)

            # Log scaler state
            profile = self.behavioral.profiles.get("live_demo_start_user")
            if profile:
                scaler = profile["scaler"]
                print(f"  Scaler mean: {scaler.mean_}")
                print(f"  Scaler std:  {scaler.scale_}")

        self.is_enrolled = True
        print("Enrollment successful!")
        return True

    def _get_face_box(self, frame):
        """Get face bounding box from MTCNN for rPPG ROI."""
        try:
            # MTCNN works with PIL RGB
            img_rgb = frame[..., ::-1].copy()
            from PIL import Image
            pil_img = Image.fromarray(img_rgb)
            boxes, probs = self.aligner.mtcnn.detect(pil_img)
            if boxes is not None and len(boxes) > 0:
                return boxes[0]  # (x1, y1, x2, y2)
        except Exception:
            pass
        return None

    def process_frame(self, frame):
        t0 = time.time()

        # 1. Face Alignment (for embedding + face detection)
        face = self.aligner.align(frame)
        if face is None:
            self.face_state = "Face not detected"
            return

        # 2. Facial Verification
        if self.is_enrolled:
            emb = self.face_auth.extract_embedding(face)
            if emb is not None:
                sim = compute_similarity(emb, self.enrolled_face_embedding)
                self.current_face_score = max(0.0, sim)
                self.face_state = (f"Recognized ({self.current_face_score:.2f})"
                                   if self.current_face_score > 0.6 else "Unrecognized")
        else:
            self.face_state = "Face Detected (Not enrolled)"

        # 3. PAD — Use detect_frame() with the FULL webcam frame
        #    This matches the training domain (full uncropped NUAA images).
        #    The old detect(face) path sent MTCNN-cropped face tensors which
        #    are out-of-distribution for the CNN trained on full images.
        pad_raw, pad_state = self.pad.detect_frame(frame)
        self.current_pad_score = pad_raw
        self.pad_availability = pad_state
        if pad_state != PAD_VALID:
            self.pad_state = f"Unavailable ({pad_state})"
        else:
            self.pad_state = (f"Live ({pad_raw:.2f})" if pad_raw > 0.5
                              else f"Spoof ({pad_raw:.2f})")

        # 4. rPPG — Pass face bounding box for proper ROI extraction
        face_box = self._get_face_box(frame)
        rppg_conf = self.rppg.process_frame(frame, face_box=face_box)
        self.current_rppg_score = rppg_conf
        buffer_full = len(self.rppg.buffer) >= self.rppg.max_frames
        if not buffer_full:
            self.rppg_state = (f"Collecting physiological signal "
                               f"({len(self.rppg.buffer)}/{self.rppg.max_frames})")
            self.rppg_availability = MODALITY_UNAVAILABLE
        else:
            self.rppg_availability = MODALITY_VALID
            if rppg_conf > 0.4:
                self.rppg_state = f"Valid Signal (Conf: {rppg_conf:.2f})"
            else:
                self.rppg_state = f"Poor Signal (Conf: {rppg_conf:.2f})"

        # 5. Keystroke Behavioral
        kb_vec = self.keys_collector.get_last_feature_vector()

        if not self.is_enrolled:
            if not self.enrollment_mode:
                self.behavioral_state = "Press E to start enrollment"
            else:
                if self.pending_kb_vec is not None:
                    self.behavioral_state = (
                        f"Typing collected ({len(self.enrolled_behavioral_samples)+1} sample(s)) "
                        "— press E again for more, or E to commit")
                elif kb_vec == "INVALID_PHRASE":
                    self.behavioral_state = "Incorrect phrase — retry"
                elif kb_vec is not None:
                    # Accumulate real enrollment samples
                    self.enrolled_behavioral_samples.append(kb_vec)
                    self.pending_kb_vec = kb_vec
                    n = len(self.enrolled_behavioral_samples)
                    self.behavioral_state = f"Sample {n} collected — press E to enroll"
                    self.last_typing_info = f"Captured: {len(kb_vec)} features"
                elif len(self.keys_collector.current_typed) > 0:
                    self.behavioral_state = "Collecting typing..."
                else:
                    self.behavioral_state = "Ready — type start"
        else:
            if kb_vec == "INVALID_PHRASE":
                self.last_typing_info = "Invalid phrase ignored"
            elif kb_vec is not None:
                payload = {
                    "session_id": "live_demo_start_user",
                    "features": kb_vec
                }

                # --- DIAGNOSTIC: Log raw behavioral evaluation ---
                self._log_behavioral_diagnostics(kb_vec, payload)

                b_eval = self.behavioral.evaluate(payload)
                b_score = b_eval.get('keystroke_score', 0.5)
                self.current_behavioral_score = b_score
                self.behavioral_availability = MODALITY_VALID
                self.behavioral_state = (f"Genuine ({b_score:.2f})" if b_score > 0.5
                                         else f"Anomalous ({b_score:.2f})")
                self.last_typing_info = f"Evaluated: {len(kb_vec)}D vec -> {b_score:.3f}"
                self._diag_logged_auth = True

        # 6. Trust Engine — with full diagnostic logging
        facial_c = 0.9 if self.is_enrolled else 0.0

        # PAD confidence: only trust if model produced a valid result
        if self.pad_availability == MODALITY_VALID:
            pad_c = 0.8
        else:
            pad_c = 0.0  # Don't penalize or reward — zero weight

        # rPPG confidence: only trust if buffer is full
        if self.rppg_availability == MODALITY_VALID:
            rppg_c = 0.7
        else:
            rppg_c = 0.0

        rppg_s = self.current_rppg_score if self.current_rppg_score is not None else 0.0
        pad_s = self.current_pad_score

        # Combine PAD + rPPG into liveness score/confidence
        # Only average components that are available
        active_liveness = []
        active_liveness_c = []
        if self.pad_availability == MODALITY_VALID:
            active_liveness.append(pad_s)
            active_liveness_c.append(pad_c)
        if self.rppg_availability == MODALITY_VALID:
            active_liveness.append(rppg_s)
            active_liveness_c.append(rppg_c)

        if active_liveness:
            live_s = sum(active_liveness) / len(active_liveness)
            live_c = sum(active_liveness_c) / len(active_liveness_c)
        else:
            live_s = 0.0
            live_c = 0.0

        # Behavioral confidence: valid only if we've actually scored a typing attempt
        if self.behavioral_availability == MODALITY_VALID:
            beh_c = 0.85
        else:
            beh_c = 0.0

        score_res = self.trust_engine.compute_trust_score(
            self.current_face_score, facial_c,
            live_s, live_c,
            self.current_behavioral_score, beh_c
        )

        self.current_trust_score = score_res.get("trust_score", 0.0)
        if self.current_trust_score >= 0.7:
            self.trust_decision = "TRUSTED"
        elif self.current_trust_score >= 0.4:
            self.trust_decision = "STEP-UP"
        else:
            self.trust_decision = "REJECTED"

        t1 = time.time()
        self.latency_ms = (t1 - t0) * 1000

        # --- DIAGNOSTIC: Log trust engine inputs (once per second) ---
        now = time.time()
        if self.is_enrolled and (now - self._last_diag_time > 1.0 or self._diag_logged_auth):
            self._last_diag_time = now
            if self._diag_logged_auth:
                self._diag_logged_auth = False
                print("\n" + "=" * 60)
                print("TRUST ENGINE DIAGNOSTIC (post-typing)")
                print("=" * 60)
            print(f"  facial_s={self.current_face_score:.4f}  facial_c={facial_c:.2f}")
            print(f"  pad_s={pad_s:.4f} ({self.pad_availability})  pad_c={pad_c:.2f}")
            print(f"  rppg_s={rppg_s:.4f} ({self.rppg_availability})  rppg_c={rppg_c:.2f}")
            print(f"  live_s={live_s:.4f}  live_c={live_c:.2f}")
            print(f"  beh_s={self.current_behavioral_score:.4f} ({self.behavioral_availability})  beh_c={beh_c:.2f}")
            weights = score_res.get("weights", {})
            print(f"  TRUST={self.current_trust_score:.4f}  decision={self.trust_decision}")
            print(f"  weights: f={weights.get('facial',0):.3f} l={weights.get('liveness',0):.3f} "
                  f"b={weights.get('behavioral',0):.3f}")
            print(f"  rPPG buffer: {len(self.rppg.buffer)}/{self.rppg.max_frames}  "
                  f"Latency: {self.latency_ms:.1f}ms")

    def _log_behavioral_diagnostics(self, kb_vec, payload):
        """Print detailed behavioral model diagnostics for forensic audit."""
        print("\n" + "-" * 60)
        print("BEHAVIORAL DIAGNOSTIC")
        print("-" * 60)
        arr = np.array(kb_vec)
        print(f"  Feature vector length: {len(kb_vec)}")
        print(f"  Feature vector: {[round(f, 4) for f in kb_vec]}")
        print(f"  shape={arr.shape} dtype={arr.dtype}")
        print(f"  mean={arr.mean():.4f} std={arr.std():.4f} min={arr.min():.4f} max={arr.max():.4f}")
        print(f"  Payload session_id: {payload.get('session_id')}")
        print(f"  Payload key used: 'features'")
        print(f"  Enrollment real samples: {self.enrollment_sample_count}")

        profile = self.behavioral.profiles.get("live_demo_start_user")
        if profile:
            scaler = profile["scaler"]
            model = profile["model"]

            X = np.array(kb_vec).reshape(1, -1)
            X_scaled = scaler.transform(X)

            raw_decision = model.decision_function(X_scaled)[0]
            prediction = model.predict(X_scaled)[0]
            mapped_score = 1 / (1 + math.exp(-raw_decision * 2.0))

            print(f"  Scaler mean: {scaler.mean_}")
            print(f"  Scaler std:  {scaler.scale_}")
            print(f"  Scaled input: {X_scaled[0]}")
            print(f"  OneClassSVM predict: {prediction}  (+1=inlier, -1=outlier)")
            print(f"  OneClassSVM decision_function: {raw_decision:.6f}")
            print(f"  Sigmoid-mapped score: {mapped_score:.6f}")
            print(f"  SVM gamma: {model._gamma}")
            print(f"  SVM nu: {model.nu}")
        else:
            print("  WARNING: No behavioral profile found for 'live_demo_start_user'!")
        print("-" * 60)

    def log_scenario(self, scenario_name):
        res = {
            "test": scenario_name,
            "timestamp": time.time(),
            "facial": {
                "state": self.face_state,
                "score": self.current_face_score
            },
            "pad": {
                "state": self.pad_state,
                "score": self.current_pad_score,
                "availability": self.pad_availability
            },
            "rppg": {
                "state": self.rppg_state,
                "score": self.current_rppg_score,
                "availability": self.rppg_availability,
                "buffer_length": len(self.rppg.buffer)
            },
            "behavioral": {
                "state": self.behavioral_state,
                "score": self.current_behavioral_score,
                "availability": self.behavioral_availability,
                "enrollment_samples": self.enrollment_sample_count
            },
            "trust": {
                "decision": self.trust_decision,
                "score": self.current_trust_score
            },
            "latency_ms": self.latency_ms
        }

        if os.path.exists(self.results_path):
            with open(self.results_path, 'r') as f:
                logs = json.load(f)
        else:
            logs = []

        logs.append(res)

        with open(self.results_path, 'w') as f:
            json.dump(logs, f, indent=2)

        print(f"\n[+] Logged scenario '{scenario_name}' to {self.results_path}")

    def draw_dashboard(self, frame):
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 290), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        y = 35
        cv2.putText(frame, "AdaptiAuth Live Demo", (20, y), self.font, 0.7,
                     (255, 255, 255), 2)
        y += 30
        cv2.putText(frame, f"FACE: {self.face_state}", (20, y), self.font, 0.5,
                     (200, 255, 200), 1)
        y += 25
        cv2.putText(frame, f"PAD : {self.pad_state}", (20, y), self.font, 0.5,
                     (200, 255, 200), 1)
        y += 25
        cv2.putText(frame, f"rPPG: {self.rppg_state}", (20, y), self.font, 0.5,
                     (200, 255, 200), 1)
        y += 25
        cv2.putText(frame, f"BEHV: {self.behavioral_state}", (20, y), self.font, 0.5,
                     (200, 255, 200), 1)
        y += 35
        cv2.putText(frame, f"TRUST DECISION: {self.trust_decision} ({self.current_trust_score:.2f})",
                     (20, y), self.font, 0.6, (0, 255, 255), 2)
        y += 35
        cv2.putText(frame, f"Latency: {self.latency_ms:.1f} ms | FPS: {int(self.cap.get(cv2.CAP_PROP_FPS))}",
                     (20, y), self.font, 0.5, (150, 150, 150), 1)
        y += 25

        # Show current typing OR last captured event
        typed = self.keys_collector.current_typed
        if typed:
            display = f"Typing: {typed}"
        elif self.last_typing_info:
            display = self.last_typing_info
        else:
            display = "Typed: (awaiting input)"
        cv2.putText(frame, display, (20, y), self.font, 0.5, (255, 255, 0), 1)

        # Instructions
        cv2.putText(frame, "[E] Enroll | [S] Log Scenario | [Q] Quit",
                     (10, frame.shape[0] - 15), self.font, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Please type: start + Enter",
                     (10, frame.shape[0] - 40), self.font, 0.5, (200, 200, 255), 1)

        return frame

    def run(self):
        self.keys_collector.start()
        self.pending_kb_vec = None
        self.enrolled_behavioral_samples = []

        print("Webcam started. Press 'E' to enroll, 'S' to log scenario, 'Q' to quit.")
        print("Enrollment supports multiple typing samples: type 'start'+Enter, "
              "then press E to add more samples or E again to commit.")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                self.process_frame(frame)
                frame = self.draw_dashboard(frame)

                cv2.imshow("AdaptiAuth Live Demo", frame)

                key = cv2.waitKey(1) & 0xFF

                if self.enrollment_mode:
                    if key == ord('e'):
                        if self.pending_kb_vec is not None:
                            # User pressed E after collecting a sample
                            # If we have enough samples, commit enrollment
                            n = len(self.enrolled_behavioral_samples)
                            if n >= 1:
                                # Commit enrollment with ALL collected samples
                                self.enroll_user(frame, self.enrolled_behavioral_samples)
                                self.enrollment_mode = False
                                self.keys_collector.reset()
                                self.behavioral_state = f"Enrolled ({n} sample(s)) — awaiting typing..."
                                self.pending_kb_vec = None
                                self.last_typing_info = f"Enrolled with {n} real sample(s)"
                            else:
                                print("Please finish typing the correct password first!")
                        else:
                            # No pending vector yet — prompt to type
                            print("Please type 'start' + Enter to provide a typing sample.")
                    elif key == ord('q'):
                        break
                else:
                    if key == ord('q'):
                        break
                    elif key == ord('e'):
                        if not self.is_enrolled:
                            self.enrollment_mode = True
                            self.enrolled_behavioral_samples = []
                            self.keys_collector.reset()
                            self.pending_kb_vec = None
                            print("Entered Enrollment Mode.")
                            print("  Type 'start' + Enter to provide a typing sample.")
                            print("  Press E again to enroll with collected sample(s).")
                            print("  Provide 3-5 samples for best behavioral accuracy.")
                    elif key == ord('s'):
                        self.scenario_counter += 1
                        scenario_name = f"LIVE_TEST_{self.scenario_counter:03d}"
                        self.keys_collector.reset()
                        self.log_scenario(scenario_name)

        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.keys_collector.stop()


if __name__ == "__main__":
    app = LiveDemoApp()
    app.run()
