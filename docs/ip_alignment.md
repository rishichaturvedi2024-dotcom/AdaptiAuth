# AdaptiAuth — IP and Novelty Traceability

This document traces the system's key innovations to ensure alignment with the intended intellectual property claims.

## 1. Continuous, Multi-Modal Trust Scoring
**Conventional Approach:** Point-in-time, binary authentication (e.g., password + 2FA at login).
**AdaptiAuth Novelty:** Authentication is a continuous background process. It does not output a binary "allow/deny" but rather a floating-point Trust Score (0-1) that decays or increases based on real-time biometric and behavioral signals.
**Traceability in Code:**
- `backend/app/services/monitoring_loop.py`: Implements the `ContinuousMonitor` that rescores active sessions periodically.
- `ml/fusion/trust_engine.py`: Implements the dynamic weighting algorithm that recalculates the score based on signal confidences.

## 2. Deep-Learning rPPG + PAD Fusion
**Conventional Approach:** Basic liveness checks using active challenges (e.g., "blink", "turn head") or simple 2D image analysis.
**AdaptiAuth Novelty:** Combines Remote Photoplethysmography (rPPG) to detect actual blood flow (physiological liveness) with a Convolutional Neural Network (CNN) Presentation-Attack Detector (PAD) for physical spoof detection.
**Traceability in Code:**
- `ml/liveness/rppg.py` and `ml/liveness/pad.py`: Implement the two distinct signals.
- `backend/app/api/routes/biometrics.py`: Defines the `/api/biometrics/liveness` endpoint that strictly requires and fuses both inputs.

## 3. Explainable AI (SHAP) for Security Scores
**Conventional Approach:** "Black-box" risk scores where the reason for a session termination or step-up challenge is opaque to the SOC analyst.
**AdaptiAuth Novelty:** Every generated Trust Score is accompanied by a SHAP (SHapley Additive exPlanations) attribution vector, explicitly detailing exactly *which* signals (and by how much) drove the score up or down.
**Traceability in Code:**
- `ml/explainability/shap_explainer.py`: Wraps the Trust Engine to compute SHAP values for the 6-dimensional input vector.
- `frontend/src/pages/SOCDashboard.jsx`: Visualizes these attributions in real-time for SOC auditability.

## 4. Zero-Trust Automated Policy Engine
**Conventional Approach:** Static timeouts or rule-based step-ups tied only to specific actions (e.g., changing a password).
**AdaptiAuth Novelty:** The Trust Score acts as a dynamic state machine. If the score dips below 0.8, a step-up challenge (FIDO2/WebAuthn) is automatically triggered without terminating the session. If it crashes below 0.5, the session is killed instantly and an alert is raised.
**Traceability in Code:**
- `backend/app/services/policy.py`: Implements the `PolicyEngine` with deterministic tier thresholds.

## 5. Federated, Privacy-Preserving Baselines
**Conventional Approach:** Sending all raw telemetry and facial data back to a central server to retrain behavioral baselines.
**AdaptiAuth Novelty:** Raw biometric data never leaves the device for retraining. The system uses a federated learning approach where the local baseline model is updated on-device, and only weight gradients are synchronized with the central server.
**Traceability in Code:**
- `backend/app/services/monitoring_loop.py` (Federated Update Interface Stub): Documents the architectural boundary for gradient-only transmission.
