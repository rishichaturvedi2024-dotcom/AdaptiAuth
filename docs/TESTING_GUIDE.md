# AdaptiAuth Comprehensive Testing Guide

Welcome to the testing guide for **AdaptiAuth** — the Risk-Adaptive Continuous Authentication Framework. 

This document provides in-depth instructions on how to test every layer of the system, from running automated unit tests to executing real-time presentation attacks against the live webcam demo.

---

## 1. Prerequisites & Environment Setup

Before running any tests, ensure your local environment is correctly configured.

### System Requirements
- Python 3.11+
- Node.js 18+
- An active webcam (for live demo scenarios)

### Setup the Virtual Environment
All backend and ML tests must be run inside the virtual environment:

```bash
# Navigate to the project root
cd adaptiauth

# Create and activate the virtual environment
python -m venv backend\venv
backend\venv\Scripts\activate   # Windows
# source backend/venv/bin/activate  # Linux/macOS

# Install all dependencies including test utilities
pip install -r requirements.txt
pip install pynput pytest pytest-asyncio pytest-cov httpx
```

---

## 2. Automated Testing (Unit & Integration)

AdaptiAuth includes a suite of automated tests verifying the FastAPI endpoints, ML stub inferences, and policy engine logic.

### Running the Test Suite
From the root of the project (with the virtual environment activated), use `pytest`:

```bash
pytest tests/ -v
```

### Coverage Report
To see how much of the backend and ML logic is covered by the tests:

```bash
pytest tests/ --cov=backend --cov=ml --cov-report=term-missing
```

**Test Structure**:
- `tests/unit/`: Contains isolated tests for specific modules (e.g., trust engine logic, policy evaluator).
- `tests/integration/`: Contains end-to-end API tests that spin up a test instance of the FastAPI application.

---

## 3. Live Webcam End-to-End Testing (Phase 10)

The most comprehensive way to test the multimodal fusion engine (Face + Liveness + Behavior) is the **Live Demo Mode**. This mode pipes your real webcam feed and keyboard inputs directly into the trust engine.

### Launching the Live Demo
1. Ensure your webcam is available and not being used by another application (like Zoom or Teams).
2. Start the script:
   ```bash
   python ml/live/live_demo.py
   ```

### Enrollment Phase
Before the engine can continuously authenticate you, it needs to know who you are.

1. Ensure your face is clearly visible.
2. Ensure the Live Demo terminal window is the active (focused) window.
3. Press `[E]` to enter Enrollment Mode. The UI will prompt: `BEHV: Ready — type start`.
4. Type the exact sequence: `start` and press `Enter`.
5. The UI will confirm the typing is collected.
6. Press `[E]` again to commit. This captures your facial embedding and initializes the behavioral SVM.

> **Note**: The behavioral model requires 200 samples in a real-world scenario. For this demo, we use synthetic noise generation around your single typing sequence to bootstrap the SVM instantly.

### Keyboard Controls
- **[E]**: Enroll (Capture face and keystroke baseline)
- **[S]**: Save/Log. Pauses the demo, asks for a scenario name, and dumps the current telemetry to `docs/live_demo_results.json`.
- **[Q]**: Quit the demo gracefully.

---

## 4. The Seven Standardized Testing Scenarios

To fully evaluate the resilience of the AdaptiAuth system, execute these 7 scenarios during the Live Demo. Press `[S]` during each to log the results.

| Test # | Scenario | How to Execute | Expected Outcome |
|:---|:---|:---|:---|
| **TEST 1** | **Genuine Enrolled User** | Sit normally, type the password `start\n`. | **TRUSTED** (High Trust Score) |
| **TEST 2** | **Unknown Face** | Have someone else sit in front of the camera, or hold up a magazine face. | **REJECTED / STEP-UP** (Facial similarity drops significantly) |
| **TEST 3** | **Printed Photo Attack** | Hold a printed photo of your enrolled face up to the camera. | **REJECTED** (PAD spoof detected, rPPG fails to find pulse) |
| **TEST 4** | **Screen Replay Attack** | Play a video of your face on a phone and show it to the camera. | **REJECTED** (PAD detects screen moiré/reflections) |
| **TEST 5** | **Poor Lighting** | Turn off the main lights in the room. | **TRUSTED** (Assuming rPPG/Face detection can still operate; tests robustness) |
| **TEST 6** | **Insufficient Signal** | Test within the first second of sitting down (< 150 frames collected). | **TRUSTED** (System gracefully waits for physiological buffer without penalizing trust) |
| **TEST 7** | **Behavioral Anomaly** | Show genuine face, but type `start\n` erratically or extremely slowly. | **STEP-UP** (Behavioral score plummets, overriding the facial match) |

---

## 5. Testing the Full Web Application (Frontend + Backend)

If you want to test the SOC (Security Operations Center) dashboard and the frontend client simulation:

### Start the Backend
```bash
# In Terminal 1
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn backend.app.main:app --reload --port 8000
```

### Start the Frontend
```bash
# In Terminal 2
cd frontend
npm install
npm run dev
```

### Testing the UI flows:
1. Open `http://localhost:5173` in your browser.
2. Click **Launch Login Demo**.
3. Allow webcam access. You will see simulated/real telemetry streaming.
4. Open a second tab to `http://localhost:5173/dashboard` to view the **Live SOC Dashboard**. 
5. As you simulate attacks or poor conditions in the first tab, watch the Trust Score dip and the SHAP explainability graphs shift in real-time on the SOC dashboard.

---

## 6. API Testing & Documentation (FastAPI)

For developers wanting to test specific endpoints or the zero-trust policy engine programmatically:

1. Start the backend (`uvicorn backend.app.main:app --reload`).
2. Navigate to `http://localhost:8000/docs`.
3. You will see the interactive **Swagger UI**.
4. From here, you can execute `POST` requests to `/api/v1/auth/evaluate` by passing mock JSON payloads representing different multimodal signals (e.g., passing a high spoof probability to see how the policy engine responds).

---

## 7. Troubleshooting

- **Webcam not turning on**: Ensure no other background process (OBS, Zoom, Discord) is locking the camera. Change the `cv2.VideoCapture(0)` index to `1` or `2` in `ml/live/live_demo.py` if you have multiple cameras.
- **Keystrokes not registering**: Ensure the terminal window running the Python script has focus, and that your OS allows terminal applications to monitor input (macOS requires Accessibility permissions for terminal).
- **ModuleNotFoundError**: Ensure the virtual environment is activated and `pip install -e .` or `pip install -r requirements.txt` was run from the project root.
