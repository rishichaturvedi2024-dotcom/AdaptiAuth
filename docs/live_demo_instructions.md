# Live Webcam Integration Test (Phase 10)

This guide documents the implementation and execution procedure for the Live Webcam Demonstration of the complete AdaptiAuth Phase 10 ML pipeline.

## A. Exact Launch Command
Start the live dashboard from the project root using the virtual environment:
```bash
backend\venv\Scripts\python.exe ml/live/live_demo.py
```

## B. Dependency Installation Command
If `pynput` is not already installed for background keystroke capturing:
```bash
backend\venv\Scripts\python.exe -m pip install pynput
```

## C. Exact Demo Enrollment Procedure
> [!WARNING]
> **Live Demo Enrollment** differs significantly from the Phase 10 research protocol (which requires 200 CMU dataset samples). This script uses synthetic noise generation around a single typing sequence purely to instantiate the SVM for the demonstration.

1. Ensure your face is clearly visible to the webcam.
2. Ensure the Live Demo window has focus.
3. Press `[E]` to enter Live Demo Enrollment mode. The UI will show: `BEHV: Ready — type start`.
4. Type the live demo password exactly: `start` followed by the `Enter` key.
5. The UI will confirm `BEHV: Typing collected`.
6. Press `[E]` again to commit the enrollment. The UI will confirm your face is registered and bootstrap the Behavioral SVM.

## D. Keyboard Interaction Instructions
- **Typing**: The background listener continuously monitors for the 6-key sequence `start\n`. Once the sequence is matched, it computes the 16-feature timing vector and sends it to the pipeline.
- **[E] Enroll**: Initializes the session using the current frame and the most recently typed sequence.
- **[S] Save/Log Scenario**: Pauses to ask for a scenario name in the terminal (e.g., "TEST 1"), then writes the current comprehensive JSON state (face, pad, rppg, behavior, trust, latency) to `docs/live_demo_results.json`.
- **[Q] Quit**: Safely exits the webcam loop and stops the keyboard listener.

## E. All 7 Test Scenarios

Execute these sequentially. Press `[S]` during each test to record the results.

1. **TEST 1: Genuine enrolled user**
   - *Action*: Sit normally, type the password.
   - *Expect*: Trusted.
2. **TEST 2: Unknown/unrecognized face**
   - *Action*: Have a different person sit in front of the camera, or hold a face from a magazine.
   - *Expect*: Step-Up / Rejected (Face Similarity drops).
3. **TEST 3: Printed photograph attack**
   - *Action*: Hold a printed photo of the enrolled face up to the camera.
   - *Expect*: Step-Up / Rejected (PAD Spoof detected, no rPPG).
4. **TEST 4: Phone/screen replay attack**
   - *Action*: Play a video of the enrolled face on a smartphone.
   - *Expect*: Step-Up / Rejected (PAD Spoof detected).
5. **TEST 5: Genuine user with poor lighting**
   - *Action*: Turn off main lights.
   - *Expect*: Trusted / Step-Up (Depending on if face detection/PAD suffers, but rPPG might still extract pulse if sufficient).
6. **TEST 6: Genuine user without sufficient rPPG signal**
   - *Action*: Sit in front of the camera for 1 second (buffer < 150 frames).
   - *Expect*: Trusted (rPPG handles collection gracefully without penalizing the overall trust).
7. **TEST 7: Genuine face + different typing pattern**
   - *Action*: Face the camera, but type the password extremely slowly or erratically.
   - *Expect*: Step-Up / Challenge (Behavioral anomaly detected, overriding facial match).

## F. Expected UI States
- **FACE**: `Recognized (Score)` or `Unrecognized`
- **PAD**: `Live (Score)` or `Spoof (Score)`
- **rPPG**: `Collecting physiological signal (x/150)` or `Valid Signal (BPM: X, SNR: Y)` or `Poor Signal`
- **BEHAVIOR**: `Press E to start enrollment`, `Ready — type start`, `Collecting typing...`, `Typing collected`, `Genuine (Score)`, or `Anomalous (Score)`
- **TRUST DECISION**: `TRUSTED`, `STEP-UP`, or `REJECTED`

## G. Known Limitations
- The behavioral SVM is forcefully bootstrapped using synthetic noise for the live demo. This is solely because capturing 200 genuine keyboard sequences interactively is impractical. 
- The PyTorch PAD CNN is optimized for static RGB presentation attacks. It may not reliably detect advanced 3D silicone masks.
- The `pynput` listener captures system-wide keystrokes; ensure you only type the password when intending to authenticate.
